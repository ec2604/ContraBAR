
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from utils import helpers as utl
from models.encoder import RNNCPCEncoder
from models.cpc_modules import MLP, actionGRU, statePredictor
from utils.helpers import get_task_dim, generate_predictor_input, relabel_func
from utils.storage_cpc import RolloutStorage
from collections import namedtuple


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CPCStats = namedtuple('cpc_stats', ['cpc_loss', 'hidden_states', 'fraction_examples_reward_seen',
                                    'fraction_trajectories_reward_seen', 'num_sampled', 'tasks'])
evaluationStats = namedtuple('evaluation_stats', ['loss', 'predictions'])


class contrabarCPC:
    """
    CPC Module of contrabar:
    - has an encoder
    - can compute the BCE loss
    - can update the CPC module
    """

    def __init__(self, args, logger, get_iter_idx):

        self.args = args
        self.logger = logger
        self.get_iter_idx = get_iter_idx
        self.task_dim = get_task_dim(self.args)  # if self.args.decode_task else None
        # self.num_tasks = get_num_tasks(self.args) if self.args.decode_task else None

        # initialise the encoder
        self.encoder = self.initialise_encoder()
        if self.args.with_action_gru:
            self.action_gru = actionGRU(input_dim=self.args.action_embedding_size, hidden_dim=self.args.latent_dim).to(
                device)
            self.action_gru_encoder = utl.FeatureExtractor(self.args.action_dim, self.args.action_embedding_size,
                                                           F.elu).to(device)
            self.action_gru_state_encoder = utl.FeatureExtractor(*self.args.state_dim, self.args.latent_dim, F.elu).to(device)

        self.lookahead_factor = self.args.lookahead_factor if self.args.lookahead_factor else 1
        z_dim = self.args.reward_embedding_size + self.args.state_embedding_size  # self.encoder.state_encoder.output_size#
        if not self.args.with_action_gru:
            z_dim += self.args.action_embedding_size
        if len(self.args.encoder_layers_before_gru) > 0:
            z_dim = self.args.encoder_layers_before_gru[-1]

        if self.args.with_action_gru:
            cpc_clf = [MLP(
                z_dim + 2*self.args.latent_dim)
                for _ in range(1)]
        else:
            cpc_clf = [MLP(
                z_dim + self.args.latent_dim) for _ in range(self.lookahead_factor)]
        self.mlp = nn.ModuleList(cpc_clf).to(device)
        self.cpc_loss_func = torch.nn.CrossEntropyLoss()
        if self.args.evaluate_representation:
            self.representation_evaluator = statePredictor(self.args.latent_dim + 3, 1).to(device)
            self.representation_evaluation_loss = nn.MSELoss()
            self.representation_evaluator_optimizer = torch.optim.Adam(self.representation_evaluator.parameters(),
                                                                       lr=self.args.evaluator_lr)
        # initialise rollout storage for the CPC update
        # (this differs from the data that the on-policy RL algorithm uses)
        self.rollout_storage = RolloutStorage(num_processes=self.args.num_processes,
                                              max_trajectory_len=self.args.max_trajectory_len,
                                              zero_pad=True,
                                              max_num_rollouts=self.args.num_trajs_representation_learning_buffer,
                                              state_dim=self.args.state_dim,
                                              action_dim=self.args.action_dim,
                                              representation_learner_buffer_add_thresh=self.args.representation_learner_buffer_add_thresh,
                                              task_dim=self.task_dim,
                                              underlying_state_dim=self.args.underlying_state_dim
                                              )
        cpc_params = [*self.encoder.parameters()] + [*self.mlp.parameters()]
        if self.args.with_action_gru:
            cpc_params += [*self.action_gru.parameters()]
            cpc_params += [*self.action_gru_encoder.parameters()]
            cpc_params += [*self.action_gru_state_encoder.parameters()]
        self.cpc_optimizer = torch.optim.Adam(cpc_params,
                                              lr=self.args.lr_representation_learner)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.cpc_optimizer, milestones=[1200], gamma=0.5)

    def initialise_encoder(self):
        """ Initialises and returns an RNN encoder """
        encoder = RNNCPCEncoder(
            args=self.args,
            layers_before_gru=self.args.encoder_layers_before_gru,
            hidden_size=self.args.latent_dim,  # self.args.encoder_gru_hidden_size,
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

    def sample_negatives(self, z_batch, negative_sampling_factor, trajectory_len, tasks, metric='l2'):
        z_batch = z_batch.permute([1, 0, 2])
        cdist = torch.cdist(z_batch.reshape(-1, z_batch.shape[-1]), z_batch.reshape(-1, z_batch.shape[-1]))
        task_dist = torch.cdist(tasks, tasks)
        x, y = torch.where(task_dist == 0)
        x = ((x * (z_batch.shape[1])).reshape(-1, 1) + torch.arange(z_batch.shape[1]).reshape(1, -1).to(
            device)).reshape(-1, 1).repeat(1, z_batch.shape[1]).reshape(-1, 1)
        y = ((y * z_batch.shape[1]).reshape(-1, 1) + torch.arange(z_batch.shape[1]).to(device)).unsqueeze(1).expand(
            y.shape[0], z_batch.shape[1], z_batch.shape[1]).reshape(-1, 1)
        idx = x * (cdist.shape[0]) + y
        orig_shape = cdist.shape
        cdist = cdist.reshape(-1, )
        cdist[idx] = -2
        cdist = cdist.reshape(orig_shape)


        numerator = (cdist > 0)
        denom = (cdist > 0).sum(dim=-1).view(cdist.shape[0], 1)
        if torch.any(denom == 0):
            return False
        obs_neg = z_batch.reshape(-1, z_batch.shape[-1])[
            torch.multinomial(numerator / denom, negative_sampling_factor)]
        return obs_neg.reshape(z_batch.shape[0], z_batch.shape[1], negative_sampling_factor, z_batch.shape[-1]).permute(
            [1, 0, 2, 3]), cdist

    def sample_negatives_2(self, z_batch, negative_sampling_factor, trajectory_len, tasks):
        z_batch = z_batch.permute([1, 0, 2])
        dim_cdist = z_batch.reshape(-1, z_batch.shape[-1]).shape[0]
        block = torch.ones((z_batch.shape[1], z_batch.shape[1]), dtype=torch.int8) * -1
        cdist = torch.block_diag(*[block for _ in range(dim_cdist // z_batch.shape[1])])

        numerator = (cdist > -1)
        denom = (cdist > -1).sum(dim=-1).view(cdist.shape[0],
                                              1)
        if torch.any(denom == 0):
            return False
        obs_neg = z_batch.reshape(-1, z_batch.shape[-1])[
            torch.multinomial(numerator / denom, negative_sampling_factor, replacement=True)]
        return obs_neg.reshape(z_batch.shape[0], z_batch.shape[1], negative_sampling_factor, z_batch.shape[-1]).permute(
            [1, 0, 2, 3]), cdist

    def sample_negative_rewards(self, next_obs, pos, actions, rewards, negative_sampling_factor,
                                relabel_func, tasks):
        rewards_modified = relabel_func(pos, negative_sampling_factor, rewards,tasks)
        z_negatives = [self.encoder.embed_input(actions, next_obs, rew_mod[..., -1], with_actions=False if self.args.with_action_gru else True) for rew_mod in  torch.split(rewards_modified, 1, dim=-2)]
        z_negatives = torch.stack(z_negatives, dim=-2)
        return z_negatives, None

    def relabel_visual_reward(self, z_batch, next_obs, rewards, actions, underlying_states):
        z_batch = z_batch.permute([1, 0, 2])
        dim_cdist = z_batch.reshape(-1, z_batch.shape[-1]).shape[0]
        block = torch.ones((z_batch.shape[1], z_batch.shape[1]), dtype=torch.int8) * -1
        cdist = torch.block_diag(*[block for _ in range(dim_cdist // z_batch.shape[1])])

        top_strip = next_obs[0, 0, 0, :20].clone()
        blank_strip = torch.zeros_like(top_strip).to(device)

        next_obs_negatives = next_obs.clone()
        r = (underlying_states[..., 0] ** 2 + underlying_states[..., 1] ** 2) ** 0.5
        theta = torch.atan2(underlying_states[..., 1], (underlying_states[..., 0]))
        mask = (theta > -np.pi) * (theta < 0) * (r > (0.2 - 0.05)) * (r < (0.2 + 0.05))
        bit_mask = torch.randn(mask.shape, device=torch.device("cuda")) < 0.5
        mask = mask * bit_mask
        next_obs_negatives[:, :, 2, :20, :][(mask) * (rewards > 0).squeeze(-1), :20, :] = top_strip.unsqueeze(0)
        next_obs_negatives[:, :, 2, :20, :][(mask) * (rewards < 1).squeeze(-1), :20, :] = blank_strip.unsqueeze(0)
        z_negatives = self.encoder.embed_input(actions=actions, states=next_obs_negatives, rewards=rewards,
                                               with_actions=False if self.args.with_action_gru else True).unsqueeze(-2)
        return z_negatives, cdist

    def compute_cpc_loss(self, batch=None):
        """ Returns the CPC loss """

        if not self.rollout_storage.ready_for_update() and batch is None:
            return 0, 0, 0

        # get a mini-batch
        # prev_obs will be of size: max trajectory len x num trajectories x dimension of observations

        # Batch can be sampled from rollout storage or directly given (as in evaluation)
        if batch is None:
            prev_obs, next_obs, actions, rewards, tasks, underlying_states, \
            trajectory_lens, num_sampled = self.rollout_storage.get_batch(
                batchsize=self.args.representation_learner_batch_num_trajs)
        else:
            prev_obs, next_obs, actions, rewards, tasks, underlying_states, trajectory_lens \
                = batch
            trajectory_lens = np.array(trajectory_lens)
            num_sampled = np.array([0])

        # If true augments the observations with random RGB shifts
        if self.args.augment_z:
            obs_list = list(torch.split(next_obs,dim=2,split_size_or_sections=[3,next_obs.shape[2]-3]))
            orig_shape = obs_list[0].shape
            rgb = obs_list[0].view(-1, *orig_shape[2:]) / 255
            rgb = kornia.augmentation.RandomRGBShift(p=1.)(rgb) * 255
            obs_list[0] = rgb.view(orig_shape)
            next_obs_new = torch.cat(obs_list,dim=2)

        # pass through encoder (outputs will be: (max_traj_len+1) x number of rollouts x latent_dim -- includes the prior!)

        z_batch = self.encoder.embed_input(actions=actions, states=next_obs_new if self.args.augment_z else next_obs, rewards=rewards,
                                           with_actions=False if self.args.with_action_gru else True)
        # Relevant for sparse rewards, calculates how many rewards seen so far at each point
        seen_reward = ((rewards > 0).cumsum(dim=0) > 0).long().permute([1, 2, 0])[..., :-1]

        # Encode each partial history
        hidden_states = self.encoder(actions=actions,
                                     states=next_obs,
                                     rewards=rewards,
                                     hidden_state=None,
                                     return_prior=False,
                                     detach_every=self.args.tbptt_stepsize if hasattr(self.args,
                                                                                      'tbptt_stepsize') else None,
                                     )
        if self.args.sampling_method == 'fast':
            negatives = self.sample_negatives_2(z_batch, self.args.negative_factor, trajectory_lens, tasks)
        elif self.args.sampling_method == 'precise':
            negatives = self.sample_negatives(z_batch, self.args.negative_factor, trajectory_lens, tasks)
        elif self.args.sampling_method == 'negative_rewards':
            negatives = self.sample_negative_rewards(next_obs, next_obs[...,:2], actions, rewards, self.args.negative_factor,
            relabel_func, tasks)
        else:
            negatives = False
        if negatives == False:
            return None
        z_negatives, z_dist = negatives

        # Initialize variables
        cpc_loss = None
        z_a_gru_pos = None
        z_a_gru_neg = None

        if self.args.with_action_gru:
            # Precalculate action indices for lookahead
            indices = torch.arange(0, self.lookahead_factor, device=device) + torch.arange(
                trajectory_lens.max() - self.lookahead_factor, device=device).view(-1, 1)

            # Gather all encoded actions at relevant indices
            encoded_actions = self.action_gru_encoder(actions)
            a_for_gru = torch.gather(
                encoded_actions[1:, :, :].unsqueeze(1).expand(-1, trajectory_lens.max() - self.lookahead_factor, -1,
                                                              -1),
                dim=0,
                index=indices.permute([1, 0]).unsqueeze(2).unsqueeze(-1).expand(-1, -1, hidden_states.shape[1],
                                                                                self.args.action_embedding_size)).reshape(
                self.lookahead_factor, -1, self.args.action_embedding_size)

            # Get hidden states for action_gru
            hidden_for_action_gru = hidden_states[:-self.lookahead_factor, :, :].reshape(-1, hidden_states.shape[-1])

            # Generate hidden states from action_gru
            a_latent, _ = self.action_gru(a_for_gru, hidden_for_action_gru.unsqueeze(0))
            hidden_for_action_gru = hidden_states[:-self.lookahead_factor, :, :].reshape(-1, hidden_states.shape[-1]).unsqueeze(0)
            hidden_for_action_gru = torch.tile(hidden_for_action_gru, (self.lookahead_factor, 1, 1))

            states_for_action_gru = self.action_gru_state_encoder(prev_obs)[:-self.lookahead_factor, :, :].reshape(-1, self.args.latent_dim)
            a_latent, _ = self.action_gru(a_for_gru, states_for_action_gru.unsqueeze(0))
            a_latent = torch.concat([a_latent, hidden_for_action_gru], dim=-1)
            a_latent = a_latent.view(self.lookahead_factor, trajectory_lens.max() - self.lookahead_factor,
                                     hidden_states.shape[1], -1)

            # Create positive and negative observations by concatenating embedded transition tuples with action_gru
            # hidden states
            z_a_gru_pos = [
                torch.cat([z_batch[1:-self.lookahead_factor, ...], a_latent[i, :-1, ...]#,
                           ], dim=-1).permute([1, 0, -1])
                for i in range(self.lookahead_factor)]

            z_a_gru_neg = [torch.cat([z_negatives[1:-self.lookahead_factor, ...],
                                      torch.unsqueeze(a_latent[i, :-1, ...], 2).expand(-1, -1,
                                                                                       self.args.negative_factor, -1),
                                      ],
                                     dim=-1).permute([1, 0, 2, 3]) for i in range(self.lookahead_factor)]
        else:
            z_a_gru_pos = [
                torch.cat([z_batch[i:, ...], hidden_states[:-i, ...]], dim=-1).permute([1, 0, -1]) for i
                in range(1, self.lookahead_factor + 1)]
            z_a_gru_neg = [torch.cat([z_negatives[i:, ...],
                                      torch.unsqueeze(hidden_states[:-i, ...], 2).expand(-1, -1,
                                                                                         self.args.negative_factor,
                                                                                         -1)],
                                     dim=-1).permute(
                [1, 0, 2, 3]) for i in range(1, self.lookahead_factor + 1)]
        if self.args.with_action_gru:
            preds = [
                torch.cat([self.mlp[0](z_a_gru_pos[i]), self.mlp[0](z_a_gru_neg[i]).squeeze(-1)], dim=-1).view(-1,
                                                                                                               1 + self.args.negative_factor)
                for i in range(self.lookahead_factor)]
        else:
            preds = [
                torch.cat([self.mlp[i](z_a_gru_pos[i]), self.mlp[i](z_a_gru_neg[i]).squeeze(-1)], dim=-1).view(
                    -1,
                    1 + self.args.negative_factor)
                for i in range(self.lookahead_factor)]
        preds = torch.cat(preds, dim=0)
        targets = torch.zeros((preds.shape[0]), dtype=torch.long, device=device)
        cpc_loss = self.cpc_loss_func(preds, targets)

        fraction_examples_reward_seen = seen_reward.sum() / z_a_gru_pos[0].shape[1]
        fraction_trajectories_reward_seen = (seen_reward.sum(dim=[1, 2]) > 0).sum() / seen_reward.shape[0]

        cpc_train_stats = CPCStats(cpc_loss, hidden_states.detach().clone(),
                                   fraction_examples_reward_seen,
                                   fraction_trajectories_reward_seen, num_sampled,
                                   tasks)
        return cpc_train_stats

    def update_cpc(self):
        for i in range(self.args.num_representation_learner_updates):
            cpc_train_stats = None
            while cpc_train_stats is None:
                cpc_train_stats = self.compute_cpc_loss()
            self.cpc_optimizer.zero_grad()
            cpc_train_stats.cpc_loss.backward()
            # clip gradients
            if self.args.encoder_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.encoder_max_grad_norm)
                if self.args.with_action_gru:
                    nn.utils.clip_grad_norm_(self.action_gru.parameters(), self.args.encoder_max_grad_norm)
            # update
            self.cpc_optimizer.step()
            self.scheduler.step()
        return cpc_train_stats

    ## Classification
    # def calc_latent_evaluator_loss(self, tasks, hidden_states):
    #     r = 1.
    #     hidden_states_for_belief = hidden_states[50:].detach()
    #     # r = 0.24
    #     num_points_semicircle = 50
    #     angles = np.linspace(0, np.pi, num=num_points_semicircle)
    #     # angles = np.linspace(np.pi / 2, np.pi / 2 + np.pi, num=num_points_semicircle)
    #     x, y = r * np.cos(angles), r * np.sin(angles)
    #     x_task, y_task = r * torch.cos(tasks[:, 0]), r * torch.sin(tasks[:, 0])
    #     task_points = torch.stack([x_task, y_task], dim=-1)
    #     # task_points = tasks
    #     circ_points = torch.from_numpy(np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])).to(device)
    #     circle_labels = (torch.norm(
    #         (task_points.unsqueeze(1).repeat(1, num_points_semicircle, 1) - circ_points.unsqueeze(0)),
    #         # dim=-1) <= 0.05).float().unsqueeze(1).repeat(1, hidden_states_for_belief.shape[0], 1)
    #         dim=-1) <= 0.2).float().unsqueeze(1).repeat(1, hidden_states_for_belief.shape[0], 1)
    #     evaluation_loss = self.representation_evaluation_loss(
    #         self.representation_evaluator(hidden_states_for_belief).view(-1, num_points_semicircle),
    #         circle_labels.view(-1, num_points_semicircle))
    #     return evaluation_loss

    # regression - reacher
    # def calc_latent_evaluator_loss(self, tasks, hidden_states):
    #     # r = 1.
    #     hidden_states_for_belief = hidden_states[50:].detach()
    #     r = 0.24
    #     num_points_semicircle = 50
    #     angles = np.linspace(np.pi / 2, np.pi / 2 + np.pi, num=num_points_semicircle)
    #     x, y = r * np.cos(angles), r * np.sin(angles)
    #     x_task, y_task = r * torch.cos(tasks[:, 0]), r * torch.sin(tasks[:, 0])
    #     task_points = torch.stack([x_task, y_task], dim=-1)
    #     # task_points = tasks
    #     circ_points = torch.from_numpy(np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])).to(device)
    #     circle_labels = (torch.norm(
    #         (task_points.unsqueeze(1).repeat(1, num_points_semicircle, 1) - circ_points.unsqueeze(0)),
    #         dim=-1) <= 0.05).float().unsqueeze(1).repeat(1, hidden_states_for_belief.shape[0], 1)
    #     # dim=-1) <= 0.2).float().unsqueeze(1).repeat(1, hidden_states_for_belief.shape[0], 1)
    #     evaluation_loss = self.representation_evaluation_loss(
    #         self.representation_evaluator(hidden_states_for_belief).view(-1, num_points_semicircle),
    #         circle_labels.view(-1, num_points_semicircle))
    #     return evaluation_loss

    # # #Regression
    # def calc_latent_evaluator_loss(self, tasks, hidden_states):
    #     r = 1.
    #     num_points_semicircle = 50
    #     angles = np.linspace(0, np.pi, num=num_points_semicircle)
    #     x, y = r * np.cos(angles), r * np.sin(angles)
    #     x_task, y_task = r * torch.cos(tasks[:, 0]), r * torch.sin(tasks[:, 0])
    #     task_points = torch.stack([x_task, y_task], dim=-1)
    #     # task_points = tasks
    #     circ_points = torch.from_numpy(np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])).to(device)
    #     circle_labels = (torch.norm(
    #         (task_points.unsqueeze(1).repeat(1, num_points_semicircle, 1) - circ_points.unsqueeze(0)),
    #         dim=-1) <= 0.3).float().unsqueeze(1).repeat(1, hidden_states_for_belief.shape[0], 1)
    #     # dim=-1) <= 0.2).float().unsqueeze(1).repeat(1, hidden_states_for_belief.shape[0], 1)
    #     evaluation_loss = self.representation_evaluation_loss(
    #         self.representation_evaluator(hidden_states_for_belief).view(-1, num_points_semicircle),
    #         circle_labels.view(-1, num_points_semicircle))
    #     return evaluation_loss

    # #Regression
    def calc_latent_evaluator_loss(self, tasks, hidden_states, train=False):
        predictor_input, labels, _ = generate_predictor_input(hidden_states, tasks, train=train)
        evaluation_loss = self.representation_evaluation_loss(
            self.representation_evaluator(predictor_input), labels)
        return evaluation_loss

    def update_latent_evaluator(self, tasks, hidden_states):

        evaluation_loss = self.calc_latent_evaluator_loss(tasks, hidden_states, train=True)
        self.representation_evaluator_optimizer.zero_grad()
        evaluation_loss.backward()
        self.representation_evaluator_optimizer.step()
        return evaluation_loss
