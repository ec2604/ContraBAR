import warnings

import gym
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn

from models.encoder import RNNCPCEncoder
from models.cpc_modules import MLP, actionGRU, statePredictor, CPCMatrix
from utils.helpers import get_task_dim, get_num_tasks, get_pos_grid
from utils.storage_vae import RolloutStorageVAE
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()


class VaribadCPC:
    """
    CPC Module of VariBAD:
    - has an encoder
    - can compute the BCE loss
    - can update the CPC module
    """

    def __init__(self, args, logger, get_iter_idx):

        self.args = args
        self.logger = logger
        self.get_iter_idx = get_iter_idx
        self.task_dim = get_task_dim(self.args)# if self.args.decode_task else None
        self.num_tasks = get_num_tasks(self.args) if self.args.decode_task else None

        # initialise the encoder
        self.encoder = self.initialise_encoder()
        # self.action_gru = actionGRU(input_dim=self.args.action_embedding_size, hidden_dim=self.args.latent_dim).to(device)
        self.lookahead_factor = self.args.lookahead_factor if self.args.lookahead_factor else 1
        # cpc_clf = [CPCMatrix(self.args.latent_dim, self.args.reward_embedding_size + self.args.state_embedding_size + self.args.action_embedding_size)
        #            for _ in range(1)]
        cpc_clf = [MLP(self.args.reward_embedding_size + self.args.state_embedding_size + self.args.action_embedding_size + self.args.latent_dim) for i in range(1)]
        # cpc_clf = [MLP(4)]
            #(2*(self.args.reward_embedding_size + self.args.state_embedding_size + 2))) for i in range(self.lookahead_factor)]
        self.mlp = nn.ModuleList(cpc_clf).to(device)
        self.cpc_loss_func = nn.BCEWithLogitsLoss()
        self.reward_predictor = statePredictor(self.args.latent_dim, self.task_dim).to(device)
        self.reward_predictor_loss = nn.MSELoss()
        self.reward_predictor_optimizer = torch.optim.Adam(self.reward_predictor.parameters(), lr=3e-4)
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
                                              #+ [*self.action_gru.parameters()], lr=self.args.lr_vae)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.cpc_optimizer, milestones=[100], gamma=0.1)

    def initialise_encoder(self):
        """ Initialises and returns an RNN encoder """
        encoder = RNNCPCEncoder(
            args=self.args,
            layers_before_gru=self.args.encoder_layers_before_gru,
            hidden_size=self.args.latent_dim, #self.args.encoder_gru_hidden_size,
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

    def sample_negatives(self, z_batch, negative_sampling_factor, trajectory_len, tasks):
        z_batch = z_batch.permute([1,0, 2])
        cdist = torch.cdist(z_batch.reshape(-1, z_batch.shape[-1]), z_batch.reshape(-1, z_batch.shape[-1]))
        task_dist = torch.cdist(tasks, tasks)
        x, y = torch.where(task_dist == 0)
        x = ((x*(z_batch.shape[1])).reshape(-1,1) + torch.arange(z_batch.shape[1]).reshape(1,-1).to(device)).reshape(-1,1).repeat(1,z_batch.shape[1]).reshape(-1,1)
        y = ((y * z_batch.shape[1]).reshape(-1, 1) + torch.arange(z_batch.shape[1]).to(device)).unsqueeze(1).expand(y.shape[0],z_batch.shape[1],z_batch.shape[1]).reshape(-1,1)
        idx = x*(cdist.shape[0]) + y
        orig_shape = cdist.shape
        cdist = cdist.reshape(-1,)
        cdist[idx] = -1
        cdist = cdist.reshape(orig_shape)

        # for i in range(z_batch.shape[0]):
        #     for j in torch.where(task_dist[i,:] == 0.)[0]:
        #         cdist[i*z_batch.shape[1]:(i+1)*z_batch.shape[1],j*z_batch.shape[1]:(j+1)*z_batch.shape[1]] = -1
        numerator = (cdist > 0)#(cdist > cdist.min(dim=-1).values.view(cdist.shape[0], 1))
        denom = (cdist > 0).sum(dim=-1).view(cdist.shape[0],1)#(cdist > cdist.min(dim=-1).values.view(cdist.shape[0], 1)).sum(dim=-1).view(cdist.shape[0], 1)
        if torch.any(denom == 0):
            return False
        obs_neg = z_batch.reshape(-1, z_batch.shape[-1])[
            torch.multinomial(numerator / denom, negative_sampling_factor)]
        return obs_neg.reshape(z_batch.shape[0], z_batch.shape[1], negative_sampling_factor, z_batch.shape[-1]).permute([1, 0, 2, 3]), cdist

    def sample_negatives_2(self, z_batch, negative_sampling_factor, trajectory_len, tasks):
        z_batch = z_batch.permute([1, 0, 2])
        dim_cdist = z_batch.reshape(-1, z_batch.shape[-1]).shape[0]
        block = torch.ones((z_batch.shape[1], z_batch.shape[1]), dtype=torch.int8)*-1
        cdist = torch.block_diag(*[block for _ in range(dim_cdist // z_batch.shape[1])])



        numerator = (cdist > -1)  # (cdist > cdist.min(dim=-1).values.view(cdist.shape[0], 1))
        denom = (cdist > -1).sum(dim=-1).view(cdist.shape[0],
                                             1)  # (cdist > cdist.min(dim=-1).values.view(cdist.shape[0], 1)).sum(dim=-1).view(cdist.shape[0], 1)
        if torch.any(denom == 0):
            return False
        obs_neg = z_batch.reshape(-1, z_batch.shape[-1])[
            torch.multinomial(numerator / denom, negative_sampling_factor)]
        return obs_neg.reshape(z_batch.shape[0], z_batch.shape[1], negative_sampling_factor, z_batch.shape[-1]).permute(
            [1, 0, 2, 3]), cdist


    def compute_cpc_loss(self, update=False, pretrain_index=None, log=False,log_prefix='train', from_storage=True, batch=None):
        """ Returns the CPC loss """

        if not self.rollout_storage.ready_for_update() and from_storage:
            return 0, 0, 0

        # get a mini-batch
        if from_storage:
            vae_prev_obs, vae_next_obs, vae_actions, vae_rewards, vae_tasks, \
            trajectory_lens, num_sampled = self.rollout_storage.get_batch(batchsize=self.args.vae_batch_num_trajs)
            # vae_prev_obs will be of size: max trajectory len x num trajectories x dimension of observations
        else:
            vae_prev_obs, vae_next_obs, vae_actions, vae_rewards, vae_tasks, trajectory_lens\
                = batch
            trajectory_lens = np.array(trajectory_lens)
            num_sampled = np.array([0])

        # pass through encoder (outputs will be: (max_traj_len+1) x number of rollouts x latent_dim -- includes the prior!)
        z_batch = self.encoder.embed_input(actions=vae_actions, states=vae_next_obs, rewards=vae_rewards, with_actions=True)
        # expanded_tasks = vae_tasks.unsqueeze(0).repeat(z_batch.shape[0], 1, 1)
        # z_batch = torch.cat([vae_next_obs[...,:2], vae_rewards],dim=-1)
        seen_reward = ((vae_rewards > 0).cumsum(dim=0) > 0).long().permute([1,2,0])[...,:-1]
        hidden_states = self.encoder(actions=vae_actions,
                                                        states=vae_next_obs,
                                                        rewards=vae_rewards,
                                                        hidden_state=None,
                                                        return_prior=False,
                                                        detach_every=self.args.tbptt_stepsize if hasattr(self.args, 'tbptt_stepsize') else None,
                                                        )
        negative_factor = 15
        negs = []
        # for _ in range(negative_factor):
        #
        #     neg_theta = torch.rand(vae_tasks.shape[0]) * (np.pi)
        #     x = torch.cos(neg_theta)
        #     y = torch.sin(neg_theta)
        #     neg_tasks = torch.stack([x, y], dim=1)
            # neg_tasks = vae_tasks + torch.randn(vae_tasks.shape).to(device)
            # neg_tasks_expanded = neg_tasks.unsqueeze(0).repeat(z_batch.shape[0], 1, 1).to(device)
            # neg_rewards = -1*torch.norm(vae_next_obs[:, :, :2] - neg_tasks.unsqueeze(0).repeat(vae_next_obs.shape[0], 1, 1).to(device), dim=2).unsqueeze(-1)
            # negs.append(self.encoder.embed_input(actions=vae_actions, states=vae_next_obs, rewards=neg_rewards,
            #                                     with_actions=True).unsqueeze(2))
        # z_negatives = torch.cat(negs, dim=2)
        # z_negatives = torch.cat([z_negatives, neg_tasks_expanded.unsqueeze(2)], dim = -1)
        # z_negatives = torch.cat([vae_next_obs[...,:2], neg_rewards], dim=-1).unsqueeze(2)
        negatives = self.sample_negatives_2(z_batch, negative_factor, trajectory_lens, vae_tasks)
        if negatives == False:
            return None, None, None
        z_negatives, z_dist = negatives
        cpc_loss = None

        # indices = torch.arange(0, self.lookahead_factor,device=device) + torch.arange(trajectory_lens.max() - self.lookahead_factor,device=device).view(-1, 1)
        # vae_actions = self.encoder.action_encoder(vae_actions)
        # a_for_gru = torch.gather(vae_actions[1:, :, :].unsqueeze(1).expand(-1, trajectory_lens.max() - self.lookahead_factor, -1, -1), dim=0,
        #              index=indices.permute([1, 0]).unsqueeze(2).unsqueeze(-1).expand(-1, -1, hidden_states.shape[1], self.args.action_embedding_size)).reshape(self.lookahead_factor, -1, self.args.action_embedding_size)
        #
        # hidden_for_action_gru = hidden_states[:-self.lookahead_factor, :, :].reshape(-1, hidden_states.shape[-1])
        # a_latent, _ = self.action_gru.gru1(a_for_gru, hidden_for_action_gru.unsqueeze(0))
        # a_latent = a_latent.view(self.lookahead_factor, trajectory_lens.max() - self.lookahead_factor, hidden_states.shape[1], -1)
        # z_a_gru_pos = [torch.cat([z_batch[1:-self.lookahead_factor, ...], a_latent[i, :-1, ...]], dim=-1).permute([1,0,-1]) for i in range(self.lookahead_factor)]
        # z_a_gru_neg = [torch.cat([z_negatives[1:-self.lookahead_factor,...], torch.unsqueeze(a_latent[i, :-1, ...],2).expand(-1, -1, negative_factor, -1)],dim=-1).permute([1,0,2,3]) for i in range(self.lookahead_factor)]
   #hidden states

        z_a_gru_pos = [
            torch.cat([z_batch[i:, ...], hidden_states[:-i,...]], dim=-1).permute([1, 0, -1]) for i
            in range(1, self.lookahead_factor+1)]
        z_a_gru_neg = [torch.cat([z_negatives[i:, ...],
                                  torch.unsqueeze(hidden_states[:-i,...], 2).expand(-1, -1, negative_factor, -1)], dim=-1).permute(
            [1, 0, 2, 3]) for i in range(1, self.lookahead_factor + 1)]



        preds = [torch.cat([self.mlp[0](z_a_gru_pos[i]), self.mlp[0](z_a_gru_neg[i]).squeeze(-1)], dim=-1).view(-1, 1 + negative_factor)
                 for i in range(self.lookahead_factor)]
        # preds = [torch.cat([self.mlp[0](z_a_gru_pos[i][..., -self.args.latent_dim:],
        #                                 z_a_gru_pos[i][..., :-self.args.latent_dim]).squeeze(-1),
        #                     self.mlp[0](z_a_gru_neg[i][..., -self.args.latent_dim:],
        #                                 z_a_gru_neg[i][..., :-self.args.latent_dim]).squeeze(-1).squeeze(-1)],
        #                    dim=-1).view(-1, 1 + negative_factor)
        #          for i in range(self.lookahead_factor)]
        preds = torch.cat(preds, dim=0)

        targets = torch.zeros(preds.shape[0]).long().to(device)
        loss = torch.nn.CrossEntropyLoss()(preds, targets)

        # num_points_semicircle = 100
        # angles = np.linspace(0, np.pi, num=num_points_semicircle)
        # x, y = np.cos(angles), np.sin(angles)
        # circ_points = torch.from_numpy(np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])).to(device)
        # circle_labels = (((torch.norm((vae_tasks.unsqueeze(1).repeat(1,num_points_semicircle,1) -
        #                                  circ_points.unsqueeze(0).repeat(vae_next_obs.shape[1],1,1)),dim=2)))\
        #     .float().unsqueeze(1).repeat(1, vae_next_obs.shape[0], 1) <= 0.02).float()
        hidden_states_for_belief = hidden_states.permute([1, 0, 2]).detach().clone()
        reward_loss = self.reward_predictor_loss(self.reward_predictor(hidden_states_for_belief).view(-1,self.task_dim), vae_tasks.unsqueeze(1).expand(-1,z_batch.shape[0],-1).reshape(-1, self.task_dim))
        fraction_examples_reward_seen = seen_reward.sum() / z_a_gru_pos[0].shape[1]#.shape[0]
        fraction_trajectories_reward_seen = (seen_reward.sum(dim=[1, 2]) > 0).sum() / seen_reward.shape[0]
        cpc_loss = loss#(loss_pos + loss_neg) / 2
        if self.get_iter_idx() > 600 and log_prefix=='train':
            self.reward_predictor_optimizer.zero_grad()
            reward_loss.backward()
            self.reward_predictor_optimizer.step()
        if update:

            self.cpc_optimizer.zero_grad()
            cpc_loss.backward()
            # clip gradients
            if self.args.encoder_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.encoder_max_grad_norm)
            # update
            self.cpc_optimizer.step()
            # self.scheduler.step()
        if log:
            self.log(loss, loss, z_dist, fraction_examples_reward_seen, fraction_trajectories_reward_seen, hidden_states, num_sampled,
                     0, 0, vae_tasks, reward_loss, pretrain_index=pretrain_index, log_prefix=log_prefix)

        return cpc_loss

    def log(self, pos_loss, neg_loss, z_dist, fraction_examples_reward_seen, fraction_trajectories_reward_seen,
            hidden_states, num_sampled, pred_positive, pred_negative,tasks, reward_loss, log_prefix, pretrain_index=None):

        if pretrain_index is None:
            curr_iter_idx = self.get_iter_idx()
        else:
            curr_iter_idx = - self.args.pretrain_len * 20 + pretrain_index#self.args.num_vae_updates_per_pretrain + pretrain_index

        if (curr_iter_idx+1) % self.args.log_interval == 0:
            # belief_norm = torch.norm(hidden_states, dim=-1).reshape(-1).detach().clone()
            # sample_traj_belief_tracking = torch.cdist(hidden_states[2:100, 0, :], hidden_states[2:100, 0, :])[10, 10:].detach().clone()
            # consecutive_belief_tracking = torch.diagonal(torch.cdist(hidden_states[:, 0, :], hidden_states[:, 0, :]), offset=1)
            # tasks_dist = torch.cdist(tasks, tasks)
            # tasks_dist.fill_diagonal_(torch.max(tasks_dist))
            self.logger.add(f'cpc_losses/{log_prefix}_pos_err', pos_loss, curr_iter_idx)
            self.logger.add(f'cpc_losses/{log_prefix}_neg_err', neg_loss, curr_iter_idx)
            self.logger.add(f'cpc_losses/{log_prefix}_fraction_examples_reward_seen', fraction_examples_reward_seen, curr_iter_idx)
            self.logger.add(f'cpc_losses/{log_prefix}_fraction_trajectories_reward_seen', fraction_trajectories_reward_seen, curr_iter_idx)
            self.logger.add(f'cpc_losses/{log_prefix}_total_loss', pos_loss + neg_loss, curr_iter_idx)
            self.logger.add_hist(f'cpc_losses/{log_prefix}_z_dist', z_dist, curr_iter_idx)
            # self.logger.add(f'cpc_losses/{log_prefix}_belief_norm_mean', belief_norm.mean(), curr_iter_idx)
            # self.logger.add(f'cpc_losses/{log_prefix}_belief_norm_var', torch.std(belief_norm), curr_iter_idx)
            # self.logger.add_hist(f'cpc_losses/{log_prefix}_belief_norm', belief_norm, curr_iter_idx)
            # self.logger.add_hist(f'cpc_losses/{log_prefix}_belief_diff_sample_trajectory_hist', sample_traj_belief_tracking, curr_iter_idx)
            # self.logger.add(f'cpc_losses/{log_prefix}_belief_diff_sample_trajectory', sample_traj_belief_tracking.mean(), curr_iter_idx)
            # self.logger.add_hist(f'cpc_losses/{log_prefix}_belief_consecutive_belief', consecutive_belief_tracking, curr_iter_idx)
            self.logger.add(f'cpc_losses/{log_prefix}_avg_num_sampled', num_sampled, curr_iter_idx)
            # self.logger.add(f'cpc_losses/{log_prefix}_min_task_dist', torch.min(tasks_dist), curr_iter_idx)
            self.logger.add(f'reward_losses/{log_prefix}_reward_loss', reward_loss, curr_iter_idx)
            # fig = plt.figure()
            #plt.hist([pred_positive.detach().cpu().numpy().flatten(), pred_negative.detach().cpu().numpy().flatten()],label=['pos', 'neg'])
            # plt.legend()
            # self.logger.add_figure(f'cpc_losses/{log_prefix}_prediction_dist', fig, curr_iter_idx)