"""
Based on https://github.com/pranz24/pytorch-soft-actor-critic
"""

import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SAC(nn.Module):
    def __init__(self,
                 args,
                 policy,
                 q1_network,
                 q2_network,

                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 gamma=0.99,
                 tau=5e-3,

                 use_cql=False,
                 alpha_cql=2.,
                 entropy_alpha=0.2,
                 automatic_entropy_tuning=True,
                 alpha_lr=3e-4,
                 clip_grad_value=None
                 ):
        super().__init__()

        self.gamma = gamma
        self.tau = tau
        self.use_cql = use_cql    # Conservative Q-Learning loss
        self.alpha_cql = alpha_cql    # Conservative Q-Learning weight parameter
        self.automatic_entropy_tuning = automatic_entropy_tuning    # Wasn't tested
        self.clip_grad_value = clip_grad_value

        # q networks - use two network to mitigate positive bias
        self.qf1 = q1_network
        self.qf1_optim = Adam(self.qf1.parameters(), lr=critic_lr)

        self.qf2 = q2_network
        self.qf2_optim = Adam(self.qf2.parameters(), lr=critic_lr)

        # target networks
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)

        self.policy = policy
        self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr)

        # automatic entropy coefficient tuning
        if self.automatic_entropy_tuning:
            #self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(ptu.device)).item()
            self.target_entropy = -self.args.action_dim
            self.log_alpha_entropy = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_entropy_optim = Adam([self.log_alpha_entropy], lr=alpha_lr)
            self.alpha_entropy = self.log_alpha_entropy.exp()
        else:
            self.alpha_entropy = entropy_alpha

    def forward(self, obs):
        action, _, _, _ = self.policy(obs)
        q1, q2 = self.qf1(obs, action), self.qf2(obs, action)
        return action, q1, q2

    def act(self, obs, deterministic=False, return_log_prob=False, belief=None):
        action, mean, log_std, log_prob = self.policy(obs,
                                                      deterministic=deterministic,
                                                      return_log_prob=return_log_prob,
                                                      additional_input=belief)
        return action, mean, log_std, log_prob

    #def select_action(self, state, evaluate=False):
    #    state = torch.FloatTensor(state).to(ptu.device).unsqueeze(0)
    #    if evaluate is False:
    #        action, _, _ = self.policy.sample(state)
    #    else:
    #        _, _, action = self.policy.sample(state)
    #    return action.detach().cpu().numpy()[0]

    def select_action(self, obs):
        with torch.no_grad():
            #obs = torch.FloatTensor(obs).to(self.device)
            #obs = obs.unsqueeze(0)
            mu, _, _, _ = self.policy(
                obs, deterministic=True, return_log_prob=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        #if obs.shape[-1] != self.image_size:
        #    obs = utilss.center_crop_image(obs, self.image_size)
        with torch.no_grad():
            #obs = torch.FloatTensor(obs).to(self.device)
            #obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.policy(obs, deterministic=False, return_log_prob=False)
            return pi.cpu().data.numpy().flatten()

    def update(self,
               policy_storage,
               encoder=None,  # variBAD encoder
               rlloss_through_encoder=False,  # whether or not to backprop RL loss through encoder
               compute_cpc_loss=None  # function that can compute the VAE loss
               ):        # computation of critic loss

        vae_prev_obs, vae_next_obs, vae_actions, vae_rewards, vae_tasks, \
        trajectory_lens = self.rollout_storage.get_batch(batchsize=self.args.vae_batch_num_trajs)




        with torch.no_grad():
            #next_obs[1] = next_obs[1].float()
            next_action, _, _, next_log_prob = self.act(vae_next_obs, return_log_prob=True, belief=real_next_belief)
            #if isinstance(next_obs, list):
            #    next_belief = next_obs[1]
            #    next_action = torch.cat((next_action, next_belief), 1)
            #    next_obs = next_obs[0]

            next_action_and_belief = torch.cat((next_action, real_next_belief), 1)

            next_q1 = self.qf1_target(real_next_obs, next_action_and_belief)
            next_q2 = self.qf2_target(real_next_obs, next_action_and_belief)
            min_next_q_target = torch.min(next_q1, next_q2) - self.alpha_entropy * next_log_prob
            q_target = reward + (1. - done) * self.gamma * min_next_q_target

            #if isinstance(obs, list):
            #    obs[1] = obs[1].float()
            #    action = torch.cat((action.float(), obs[1]), 1)
            #    only_obs = obs[0]
            #else:
            #    only_obs = obs

        action_and_belief = torch.cat((action, real_belief), 1)
        q1_pred = self.qf1(real_obs, action_and_belief)
        q2_pred = self.qf2(real_obs, action_and_belief)

        qf1_loss = F.mse_loss(q1_pred, q_target)  # TD error
        qf2_loss = F.mse_loss(q2_pred, q_target)  # TD error

        # use CQL loss for offline RL (Kumar et al, 2020)
        #if self.use_cql:
        #    qf1_loss += torch.mean(self.alpha_cql *
        #                           self.estimate_log_sum_exp_q(self.qf1, obs, N=10, action_space=kwargs['action_space'])
        #                           - q1_pred)
        #    qf2_loss += torch.mean(self.alpha_cql *
        #                           self.estimate_log_sum_exp_q(self.qf2, obs, N=10, action_space=kwargs['action_space'])
        #                           - q2_pred)

        # computation of actor loss
        new_action, _, _, log_prob = self.act(real_obs, return_log_prob=True, belief=real_belief)
        #with torch.no_grad():
        #    if isinstance(obs, list):
        #        new_action = torch.cat((new_action, next_belief), 1)
        new_action_and_belief = torch.cat((new_action, real_next_belief), 1)
        min_q_new_actions = self._min_q(real_obs, new_action_and_belief)

        policy_loss = ((self.alpha_entropy * log_prob) - min_q_new_actions).mean()

        # update q networks
        self.qf1_optim.zero_grad()
        self.qf2_optim.zero_grad()
        qf1_loss.backward()
        qf2_loss.backward()
        #if self.clip_grad_value is not None:
        #    self._clip_grads(self.qf1)
        #    self._clip_grads(self.qf2)
        self.qf1_optim.step()
        self.qf2_optim.step()
        # soft update
        self.soft_target_update()

        # update policy network
        self.policy_optim.zero_grad()
        policy_loss.backward()
        #if self.clip_grad_value is not None:
        #    self._clip_grads(self.policy)
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_entropy_loss = -(self.log_alpha_entropy * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_entropy_optim.zero_grad()
            alpha_entropy_loss.backward()
            self.alpha_entropy_optim.step()

            self.alpha_entropy = self.log_alpha_entropy.exp()
            alpha_entropy_tlogs = self.alpha_entropy.clone()    # For TensorboardX logs
        else:
            alpha_entropy_loss = torch.tensor(0.).to(device) # don't forget to indent back
            alpha_entropy_tlogs = torch.tensor(self.alpha_entropy)  # For TensorboardX logs

        return {'qf1_loss': qf1_loss.item(), 'qf2_loss': qf2_loss.item(),
                'policy_loss': policy_loss.item(), 'alpha_entropy_loss': alpha_entropy_loss.item()}

    def _min_q(self, obs, action):
        q1 = self.qf1(obs, action)
        q2 = self.qf2(obs, action)
        min_q = torch.min(q1, q2)
        return min_q

    def soft_update_from_to(self, source, target, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def soft_target_update(self):
        self.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        self.soft_update_from_to(self.qf2, self.qf2_target, self.tau)

    def _clip_grads(self, net):
        for p in net.parameters():
            p.grad.data.clamp_(-self.clip_grad_value, self.clip_grad_value)

    def estimate_log_sum_exp_q(self, qf, obs, N, action_space):
        '''
            estimate log(sum(exp(Q))) for CQL objective
        :param qf: Q function
        :param obs: state batch from buffer (s~D)
        :param N: number of actions to sample for estimation
        :param action_space: space of actions -- for uniform sampling
        :return:
        '''
        batch_size = obs.shape[0]
        obs_rep = obs.repeat(N, 1)
        
        # draw actions at uniform
        random_actions = torch.FloatTensor(np.vstack([action_space.sample() for _ in range(N)]),device=device)
        random_actions = torch.repeat_interleave(random_actions, batch_size, dim=0)
        unif_a = 1 / np.prod(action_space.high - action_space.low)  # uniform density over action space

        # draw actions from current policy
        with torch.no_grad():
            policy_actions, _, _, policy_log_probs = self.act(obs_rep, return_log_prob=True)

        exp_q_unif = qf(obs_rep, random_actions) / unif_a
        exp_q_policy = qf(obs_rep, policy_actions) / torch.exp(policy_log_probs)
        log_sum_exp = torch.log(0.5 * torch.mean((exp_q_unif + exp_q_policy).reshape(N, batch_size, -1), dim=0))

        return log_sum_exp

    # # Save model parameters
    # def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
    #     if not os.path.exists('models/'):
    #         os.makedirs('models/')
    #
    #     if actor_path is None:
    #         actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
    #     if critic_path is None:
    #         critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
    #     print('Saving models to {} and {}'.format(actor_path, critic_path))
    #     torch.save(self.policy.state_dict(), actor_path)
    #     torch.save(self.critic.state_dict(), critic_path)
    #
    # # Load model parameters
    # def load_model(self, actor_path, critic_path):
    #     print('Loading models from {} and {}'.format(actor_path, critic_path))
    #     if actor_path is not None:
    #         self.policy.load_state_dict(torch.load(actor_path))
    #     if critic_path is not None:
    #         self.critic.load_state_dict(torch.load(critic_path))



