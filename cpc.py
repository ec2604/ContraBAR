import warnings

import gym
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn

from models.encoder import RNNCPCEncoder
from models.cpc_modules import MLP, actionGRU
from utils.helpers import get_task_dim, get_num_tasks
from utils.storage_vae import RolloutStorageVAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VaribadCPC:
    """
    CPC Module of VariBAD:
    - has an encoder
    - can compute the BCE loss
    - can update the CPC module
    """

    def __init__(self, args, logger, get_iter_idx, lookahead_factor):

        self.args = args
        self.logger = logger
        self.get_iter_idx = get_iter_idx
        self.task_dim = get_task_dim(self.args) if self.args.decode_task else None
        self.num_tasks = get_num_tasks(self.args) if self.args.decode_task else None

        # initialise the encoder
        self.encoder = self.initialise_encoder()
        self.action_gru = actionGRU(input_dim=self.args.action_dim, hidden_dim=self.args.latent_dim).to(device)
        self.lookahead_factor = lookahead_factor
        cpc_clf = [MLP(self.args.reward_embedding_size + self.args.state_embedding_size + self.args.latent_dim) for i in range(self.lookahead_factor)]
        self.mlp = nn.ModuleList(cpc_clf).to(device)
        self.cpc_loss_func = nn.BCEWithLogitsLoss()
        # initialise rollout storage for the VAE update
        # (this differs from the data that the on-policy RL algorithm uses)
        self.rollout_storage = RolloutStorageVAE(num_processes=self.args.num_processes,
                                                 max_trajectory_len=self.args.max_trajectory_len,
                                                 zero_pad=True,
                                                 max_num_rollouts=self.args.size_vae_buffer,
                                                 state_dim=self.args.state_dim,
                                                 action_dim=self.args.action_dim,
                                                 vae_buffer_add_thresh=self.args.vae_buffer_add_thresh,
                                                 task_dim=self.task_dim
                                                 )


        self.cpc_optimizer = torch.optim.Adam([*self.encoder.parameters()] + [*self.mlp.parameters()], lr=self.args.lr_vae)

    def initialise_encoder(self):
        """ Initialises and returns an RNN encoder """
        encoder = RNNCPCEncoder(
            args=self.args,
            layers_before_gru=self.args.encoder_layers_before_gru,
            hidden_size=self.args.latent_dim,#self.args.encoder_gru_hidden_size,
            layers_after_gru=self.args.encoder_layers_after_gru,
            latent_dim=self.args.latent_dim,
            action_dim=self.args.action_dim,
            action_embed_dim=self.args.action_embedding_size,
            state_dim=self.args.state_dim,
            state_embed_dim=self.args.state_embedding_size,
            reward_size=1,
            reward_embed_size=self.args.reward_embedding_size,
        ).to(device)
        return encoder





    def compute_cpc_loss_(self, latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, vae_actions,
                     vae_rewards, vae_tasks, trajectory_lens):
        """
        Computes the VAE loss for the given data.
        Batches everything together and therefore needs all trajectories to be of the same length.
        (Important because we need to separate ELBOs and decoding terms so can't collapse those dimensions)
        """

        num_unique_trajectory_lens = len(np.unique(trajectory_lens))

        assert (num_unique_trajectory_lens == 1) or (self.args.vae_subsample_elbos and self.args.vae_subsample_decodes)
        assert not self.args.decode_only_past

        # cut down the batch to the longest trajectory length
        # this way we can preserve the structure
        # but we will waste some computation on zero-padded trajectories that are shorter than max_traj_len
        max_traj_len = np.max(trajectory_lens)
        latent_mean = latent_mean[:max_traj_len + 1]
        latent_logvar = latent_logvar[:max_traj_len + 1]
        vae_prev_obs = vae_prev_obs[:max_traj_len]
        vae_next_obs = vae_next_obs[:max_traj_len]
        vae_actions = vae_actions[:max_traj_len]
        vae_rewards = vae_rewards[:max_traj_len]

        # take one sample for each ELBO term
        if not self.args.disable_stochasticity_in_latent:
            latent_samples = self.encoder._sample_gaussian(latent_mean, latent_logvar)
        else:
            latent_samples = torch.cat((latent_mean, latent_logvar), dim=-1)

        num_elbos = latent_samples.shape[0]
        num_decodes = vae_prev_obs.shape[0]
        batchsize = latent_samples.shape[1]  # number of trajectories

        # subsample elbo terms
        #   shape before: num_elbos * batchsize * dim
        #   shape after: vae_subsample_elbos * batchsize * dim
        if self.args.vae_subsample_elbos is not None:
            # randomly choose which elbo's to subsample
            if num_unique_trajectory_lens == 1:
                elbo_indices = torch.LongTensor(self.args.vae_subsample_elbos * batchsize).random_(0, num_elbos)    # select diff elbos for each task
            else:
                # if we have different trajectory lengths, subsample elbo indices separately
                # up to their maximum possible encoding length;
                # only allow duplicates if the sample size would be larger than the number of samples
                elbo_indices = np.concatenate([np.random.choice(range(0, t + 1), self.args.vae_subsample_elbos,
                                                                replace=self.args.vae_subsample_elbos > (t+1)) for t in trajectory_lens])
                if max_traj_len < self.args.vae_subsample_elbos:
                    warnings.warn('The required number of ELBOs is larger than the shortest trajectory, '
                                  'so there will be duplicates in your batch.'
                                  'To avoid this use --split_batches_by_elbo or --split_batches_by_task.')
            task_indices = torch.arange(batchsize).repeat(self.args.vae_subsample_elbos)  # for selection mask
            latent_samples = latent_samples[elbo_indices, task_indices, :].reshape((self.args.vae_subsample_elbos, batchsize, -1))
            num_elbos = latent_samples.shape[0]
        else:
            elbo_indices = None

        # expand the state/rew/action inputs to the decoder (to match size of latents)
        # shape will be: [num tasks in batch] x [num elbos] x [len trajectory (reconstrution loss)] x [dimension]
        dec_prev_obs = vae_prev_obs.unsqueeze(0).expand((num_elbos, *vae_prev_obs.shape))
        dec_next_obs = vae_next_obs.unsqueeze(0).expand((num_elbos, *vae_next_obs.shape))
        dec_actions = vae_actions.unsqueeze(0).expand((num_elbos, *vae_actions.shape))
        dec_rewards = vae_rewards.unsqueeze(0).expand((num_elbos, *vae_rewards.shape))

        # subsample reconstruction terms
        if self.args.vae_subsample_decodes is not None:
            # shape before: vae_subsample_elbos * num_decodes * batchsize * dim
            # shape after: vae_subsample_elbos * vae_subsample_decodes * batchsize * dim
            # (Note that this will always have duplicates given how we set up the code)
            indices0 = torch.arange(num_elbos).repeat(self.args.vae_subsample_decodes * batchsize)
            if num_unique_trajectory_lens == 1:
                indices1 = torch.LongTensor(num_elbos * self.args.vae_subsample_decodes * batchsize).random_(0, num_decodes)
            else:
                indices1 = np.concatenate([np.random.choice(range(0, t), num_elbos * self.args.vae_subsample_decodes,
                                                            replace=True) for t in trajectory_lens])
            indices2 = torch.arange(batchsize).repeat(num_elbos * self.args.vae_subsample_decodes)
            dec_prev_obs = dec_prev_obs[indices0, indices1, indices2, :].reshape((num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
            dec_next_obs = dec_next_obs[indices0, indices1, indices2, :].reshape((num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
            dec_actions = dec_actions[indices0, indices1, indices2, :].reshape((num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
            dec_rewards = dec_rewards[indices0, indices1, indices2, :].reshape((num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
            num_decodes = dec_prev_obs.shape[1]

        # expand the latent (to match the number of state/rew/action inputs to the decoder)
        # shape will be: [num tasks in batch] x [num elbos] x [len trajectory (reconstrution loss)] x [dimension]
        dec_embedding = latent_samples.unsqueeze(0).expand((num_decodes, *latent_samples.shape)).transpose(1, 0)

        if self.args.decode_reward:
            # compute reconstruction loss for this trajectory (for each timestep that was encoded, decode everything and sum it up)
            # shape: [num_elbo_terms] x [num_reconstruction_terms] x [num_trajectories]
            rew_reconstruction_loss = self.compute_rew_reconstruction_loss(dec_embedding, dec_prev_obs, dec_next_obs,
                                                                           dec_actions, dec_rewards)
            # avg/sum across individual ELBO terms
            if self.args.vae_avg_elbo_terms:
                rew_reconstruction_loss = rew_reconstruction_loss.mean(dim=0)
            else:
                rew_reconstruction_loss = rew_reconstruction_loss.sum(dim=0)
            # avg/sum across individual reconstruction terms
            if self.args.vae_avg_reconstruction_terms:
                rew_reconstruction_loss = rew_reconstruction_loss.mean(dim=0)
            else:
                rew_reconstruction_loss = rew_reconstruction_loss.sum(dim=0)
            # average across tasks
            rew_reconstruction_loss = rew_reconstruction_loss.mean()
        else:
            rew_reconstruction_loss = 0

        if self.args.decode_state:
            state_reconstruction_loss = self.compute_state_reconstruction_loss(dec_embedding, dec_prev_obs,
                                                                               dec_next_obs, dec_actions)
            # avg/sum across individual ELBO terms
            if self.args.vae_avg_elbo_terms:
                state_reconstruction_loss = state_reconstruction_loss.mean(dim=0)
            else:
                state_reconstruction_loss = state_reconstruction_loss.sum(dim=0)
            # avg/sum across individual reconstruction terms
            if self.args.vae_avg_reconstruction_terms:
                state_reconstruction_loss = state_reconstruction_loss.mean(dim=0)
            else:
                state_reconstruction_loss = state_reconstruction_loss.sum(dim=0)
            # average across tasks
            state_reconstruction_loss = state_reconstruction_loss.mean()
        else:
            state_reconstruction_loss = 0

        if self.args.decode_task:
            task_reconstruction_loss = self.compute_task_reconstruction_loss(latent_samples, vae_tasks)
            # avg/sum across individual ELBO terms
            if self.args.vae_avg_elbo_terms:
                task_reconstruction_loss = task_reconstruction_loss.mean(dim=0)
            else:
                task_reconstruction_loss = task_reconstruction_loss.sum(dim=0)
            # sum the elbos, average across tasks
            task_reconstruction_loss = task_reconstruction_loss.sum(dim=0).mean()
        else:
            task_reconstruction_loss = 0

        if not self.args.disable_kl_term:
            # compute the KL term for each ELBO term of the current trajectory
            # shape: [num_elbo_terms] x [num_trajectories]
            kl_loss = self.compute_kl_loss(latent_mean, latent_logvar, elbo_indices)
            # avg/sum the elbos
            if self.args.vae_avg_elbo_terms:
                kl_loss = kl_loss.mean(dim=0)
            else:
                kl_loss = kl_loss.sum(dim=0)
            # average across tasks
            kl_loss = kl_loss.sum(dim=0).mean()
        else:
            kl_loss = 0

        return rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss

    def sample_negatives(self, z_batch, negative_sampling_factor, trajectory_len):
        z_batch = z_batch.permute([1,0, 2])
        cdist = torch.cdist(z_batch.reshape(-1, z_batch.shape[-1]), z_batch.reshape(-1, z_batch.shape[-1]))
        for i in range(25):
            cdist[i*60:(i+1)*60,i*60:(i+1)*60] = -1
        numerator = (cdist > -1)#(cdist > cdist.min(dim=-1).values.view(cdist.shape[0], 1))
        denom = (cdist > -1).sum(dim=-1).view(cdist.shape[0],1)#(cdist > cdist.min(dim=-1).values.view(cdist.shape[0], 1)).sum(dim=-1).view(cdist.shape[0], 1)
        obs_neg = z_batch.reshape(-1, z_batch.shape[-1])[
            torch.multinomial(numerator / denom, negative_sampling_factor)]
        return obs_neg.reshape(z_batch.shape[0], z_batch.shape[1], negative_sampling_factor, z_batch.shape[-1]).permute([1, 0, 2, 3]), cdist

    def compute_cpc_loss(self, update=False, pretrain_index=None):
        """ Returns the CPC loss """

        if not self.rollout_storage.ready_for_update():
            return 0

        # get a mini-batch
        vae_prev_obs, vae_next_obs, vae_actions, vae_rewards, vae_tasks, \
        trajectory_lens = self.rollout_storage.get_batch(batchsize=self.args.vae_batch_num_trajs)
        # vae_prev_obs will be of size: max trajectory len x num trajectories x dimension of observations

        # pass through encoder (outputs will be: (max_traj_len+1) x number of rollouts x latent_dim -- includes the prior!)
        z_batch = self.encoder.embed_input(actions=vae_actions, states=vae_prev_obs, rewards=vae_rewards, with_actions=False)
        seen_reward = ((vae_rewards > 0).cumsum(dim=0) > 0).long().permute([1,2,0])[...,:-1]
        hidden_states = self.encoder(actions=vae_actions,
                                                        states=vae_next_obs,
                                                        rewards=vae_rewards,
                                                        hidden_state=None,
                                                        return_prior=True,
                                                        detach_every=self.args.tbptt_stepsize if hasattr(self.args, 'tbptt_stepsize') else None,
                                                        )
        z_negatives, z_dist = self.sample_negatives(z_batch, 1, trajectory_lens)
        indices = torch.arange(0, self.lookahead_factor,device=device) + torch.arange(trajectory_lens.max() - self.lookahead_factor,device=device).view(-1, 1)


        cpc_loss = None
        # a_for_gru = torch.gather(vae_actions[1:, :, :].unsqueeze(1).expand(-1, trajectory_lens.max() - self.lookahead_factor, -1, -1), dim=0,
        #              index=indices.permute([1,0]).unsqueeze(2).expand(-1,-1,hidden_states.shape[1]).unsqueeze(-1)).view(self.lookahead_factor,hidden_states.shape[1] * (trajectory_lens.max() - self.lookahead_factor), 1)
        # hidden_for_action_gru = hidden_states[:-self.lookahead_factor, :, :].reshape(-1, hidden_states.shape[-1])
        # a_latent, _ = self.action_gru.gru1(a_for_gru, hidden_for_action_gru.unsqueeze(0))
        # a_latent = a_latent.view(self.lookahead_factor, trajectory_lens.max() - self.lookahead_factor, hidden_states.shape[1], -1)
        # z_a_gru_pos = [torch.cat([z_batch[1:-self.lookahead_factor, ...], a_latent[i, :-1, ...]], dim=-1).permute([1,0,-1]) for i in range(self.lookahead_factor)]
        # z_a_gru_neg = [torch.cat([z_negatives[1:-self.lookahead_factor,...], torch.unsqueeze(a_latent[i, :-1, ...],2).expand(-1, -1, 1, -1)],dim=-1).permute([1,0,2,3]) for i in range(self.lookahead_factor)]
        z_a_gru_pos = [
            torch.cat([z_batch[i:, ...], hidden_states[:-i,...]], dim=-1).permute([1, 0, -1]) for i
            in range(1, self.lookahead_factor+1)]
        z_a_gru_neg = [torch.cat([z_negatives[i:, ...],
                                  torch.unsqueeze(hidden_states[:-i,...], 2).expand(-1, -1, 1, -1)], dim=-1).permute(
            [1, 0, 2, 3]) for i in range(1, self.lookahead_factor + 1)]
        pred_positive = torch.stack(
            [self.mlp[i](z_a_gru_pos[i].reshape(-1, z_a_gru_pos[i].shape[-1])).reshape(*z_a_gru_pos[i].shape[:2]) for i in
             range(self.lookahead_factor)], 1)
        pred_negative = torch.stack([self.mlp[i](z_a_gru_neg[i].reshape(-1, z_a_gru_neg[i].shape[-1])).reshape(z_a_gru_neg[i].shape[:2],
                                                                                             1,
                                                                                             -1) for i in
                                     range(self.lookahead_factor)], 1)
        pred_positive = pred_positive.view(-1, 1)
        pred_negative = pred_negative.view(-1, 1)
        pos_target = torch.ones_like(pred_positive, device=device)
        neg_target = torch.zeros_like(pred_negative, device=device)
        loss_pos = self.cpc_loss_func(pred_positive, pos_target)
        loss_neg = self.cpc_loss_func(pred_negative, neg_target)
        loss_pos_seen = (nn.BCEWithLogitsLoss(reduction='none')(pred_positive, pos_target) * seen_reward.reshape(-1, 1)).sum() / seen_reward.sum()
        loss_neg_seen = (nn.BCEWithLogitsLoss(reduction='none')(pred_negative, neg_target) * seen_reward.reshape(-1,
                                                                                                                 1)).sum() / seen_reward.sum()
        fraction_examples_reward_seen = seen_reward.sum() / pred_positive.shape[0]
        cpc_loss = (loss_pos + loss_neg) / 2
        if update:
            self.cpc_optimizer.zero_grad()
            cpc_loss.backward()
            # clip gradients
            if self.args.encoder_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.encoder_max_grad_norm)
            # update
            self.cpc_optimizer.step()
        self.log(loss_pos, loss_neg, z_dist, loss_pos_seen, loss_neg_seen, fraction_examples_reward_seen, pretrain_index=pretrain_index)

        return cpc_loss

    def log(self, pos_loss, neg_loss, z_dist, loss_pos_seen, loss_neg_seen, fraction_examples_reward_seen,
            pretrain_index=None):

        if pretrain_index is None:
            curr_iter_idx = self.get_iter_idx()
        else:
            curr_iter_idx = - self.args.pretrain_len * 20 + pretrain_index#self.args.num_vae_updates_per_pretrain + pretrain_index

        if (curr_iter_idx+1) % self.args.log_interval == 0:

            self.logger.add('cpc_losses/pos_err', pos_loss, curr_iter_idx)
            self.logger.add('cpc_losses/neg_err', neg_loss, curr_iter_idx)
            self.logger.add('cpc_losses/pos_err_reward_seen', loss_pos_seen, curr_iter_idx)
            self.logger.add('cpc_losses/neg_err_reward_seen', loss_neg_seen, curr_iter_idx)
            self.logger.add('cpc_losses/fraction_examples_reward_seen', fraction_examples_reward_seen, curr_iter_idx)
            self.logger.add('cpc_losses/total_loss', pos_loss + neg_loss, curr_iter_idx)
            self.logger.add_hist('cpc_losses/z_dist', z_dist, curr_iter_idx)
