import os
import time
import wandb
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from algorithms.a2c import A2C
from algorithms.online_storage import CPCOnlineStorage
from algorithms.ppo import PPO
from algorithms.sac import SAC
from environments.parallel_envs import make_vec_envs
from models.policy import Policy
from utils import evaluation as utl_eval
from utils import helpers as utl
from utils.tb_logger import TBLogger
from cpc import contrabarCPC
from models.encoder import ImageEncoder
from utils.storage_cpc import RolloutStorage
import datetime
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MetaLearner:
    """
    Meta-Learner class with the main training loop for contrabar.
    """

    def __init__(self, args):
        self.args = args
        utl.seed(self.args.seed, self.args.deterministic_execution)

        # calculate number of updates and keep count of frames/iterations
        self.num_updates = int(args.num_frames) // args.policy_num_steps // args.num_processes
        self.frames = 0
        self.iter_idx = -1
        wandb.init(project='varibad_cpc', config=self.args, tags=[self.args.env_name], sync_tensorboard=True,
                   name=self.args.exp_label + '_' + str(args.seed) + '_' + datetime.datetime.now().strftime('_%d:%m_%H:%M:%S'), dir='/mnt/data/erac/')
        # initialise tensorboard logger
        self.logger = TBLogger(self.args, self.args.exp_label)

        # initialise environments

        self.envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                                  gamma=args.policy_gamma, device=device,
                                  episodes_per_task=self.args.max_rollouts_per_task,
                                  normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                                  tasks=None
                                  )


        if self.args.single_task_mode:
            # get the current tasks (which will be num_process many different tasks)
            self.train_tasks = self.envs.get_task()
            # set the tasks to the first task (i.e. just a random task)
            self.train_tasks[1:] = self.train_tasks[0]
            # make it a list
            self.train_tasks = [t for t in self.train_tasks]
            # re-initialise environments with those tasks
            self.envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                                      gamma=args.policy_gamma, device=device,
                                      episodes_per_task=self.args.max_rollouts_per_task,
                                      normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                                      tasks=self.train_tasks
                                      )
            # save the training tasks so we can evaluate on the same envs later
            utl.save_obj(self.train_tasks, self.logger.full_output_folder, "train_tasks")
        else:
            self.train_tasks = None

        # calculate what the maximum length of the trajectories is
        self.args.max_trajectory_len = self.envs._max_episode_steps
        self.args.max_trajectory_len *= self.args.max_rollouts_per_task

        # get policy input dimensions
        self.args.state_dim = self.envs.observation_space.shape
        self.args.task_dim = self.envs.task_dim
        self.args.belief_dim = self.envs.belief_dim
        self.args.num_states = self.envs.num_states
        # get policy output (action) dimensions
        self.args.action_space = self.envs.action_space
        if isinstance(self.envs.action_space, gym.spaces.discrete.Discrete):
            self.args.action_dim = 1
        else:
            self.args.action_dim = self.envs.action_space.shape[0]

        # initialise CPC and policy
        self.cpc_encoder = contrabarCPC(self.args, self.logger, lambda: self.iter_idx)
        self.policy_storage = self.initialise_policy_storage()
        self.policy = self.initialise_policy()


    def initialise_policy_storage(self):
        return CPCOnlineStorage(args=self.args,
                                num_steps=self.args.policy_num_steps,
                                num_processes=self.args.num_processes,
                                state_dim=self.args.state_dim,
                                latent_dim=self.args.latent_dim,
                                task_dim=self.args.task_dim,
                                action_space=self.args.action_space,
                                hidden_size=self.args.latent_dim,
                                normalise_rewards=self.args.norm_rew_for_policy,
                                )

    def initialise_policy(self):

        # initialise policy network
        policy_net = Policy(args=self.args, pass_state_to_policy=self.args.pass_state_to_policy,
                            pass_latent_to_policy=self.args.pass_latent_to_policy,
                            pass_task_to_policy=self.args.pass_task_to_policy, dim_state=self.args.state_dim,
                            dim_latent=self.args.latent_dim, dim_task=self.args.task_dim,
                            hidden_layers=self.args.policy_layers,
                            activation_function=self.args.policy_activation_function,
                            policy_initialisation=self.args.policy_initialisation, action_space=self.envs.action_space,
                            init_std=self.args.policy_init_std, encoder=self.cpc_encoder.encoder).to(device)

        # initialise policy trainer
        if self.args.policy == 'a2c':
            policy = A2C(
                self.args,
                policy_net,
                self.args.policy_value_loss_coef,
                self.args.policy_entropy_coef,
                policy_optimiser=self.args.policy_optimiser,
                policy_anneal_lr=self.args.policy_anneal_lr,
                train_steps=self.num_updates,
                optimiser_vae=self.vae.optimizer_representation_learner,
                lr=self.args.lr_policy,
                eps=self.args.policy_eps,
            )
        elif self.args.policy == 'ppo':
            policy = PPO(self.args, policy_net, self.args.policy_value_loss_coef, self.args.policy_entropy_coef,
                         policy_optimiser=self.args.policy_optimiser, policy_anneal_lr=self.args.policy_anneal_lr,
                         train_steps=self.num_updates, optimizer_representation_learner=self.cpc_encoder.cpc_optimizer,
                         lr=self.args.lr_policy, clip_param=self.args.ppo_clip_param,
                         ppo_epoch=self.args.ppo_num_epochs, num_mini_batch=self.args.ppo_num_minibatch,
                         eps=self.args.policy_eps, use_huber_loss=self.args.ppo_use_huberloss,
                         use_clipped_value_loss=self.args.ppo_use_clipped_value_loss)
        else:
            raise NotImplementedError

        return policy

    def train(self):
        """ Main Meta-Training loop """
        start_time = time.time()
        # reset environments
        prev_state, task = utl.reset_env(self.envs, self.args)
            # insert initial observation / embeddings to rollout storage
        self.policy_storage.prev_state[0].copy_(prev_state)

        # log once before training
        with torch.no_grad():
            self.log(None, None, None, None, start_time)

        for self.iter_idx in range(self.num_updates):
            # First, re-compute the hidden states given the current rollouts (since the CPC might've changed)
            with torch.no_grad():
                hidden_state = self.encode_running_trajectory()

            # add this initial hidden state to the policy storage
            self.policy_storage.hidden_states[0].copy_(hidden_state.squeeze(0))
            # rollout policies for a few steps
            for step in range(self.args.policy_num_steps):
                # sample actions from policy
                with torch.no_grad():
                    value, action = utl.select_action_cpc(args=self.args, policy=self.policy, deterministic=False,
                                                          hidden_latent=hidden_state.squeeze(0),
                                                          state=prev_state,
                                                          task=task)

                # take step in the environment
                [next_state, task], (rew_raw, rew_normalised), done, infos = utl.env_step(self.envs, action,
                                                                                   self.args)

                done = torch.from_numpy(np.array(done, dtype=int)).to(device).float().view((-1, 1))
                # create mask for episode ends
                masks_done = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)
                # bad_mask is true if episode ended because time limit was reached
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos]).to(device)

                with torch.no_grad():
                    # compute next embedding (for next loop and/or value prediction bootstrap)
                    hidden_state = utl.update_encoding_cpc(
                        encoder=self.cpc_encoder.encoder,
                        next_obs=next_state,
                        action=action,
                        reward=rew_raw,
                        done=done,
                        hidden_state=hidden_state)

                # before resetting, update the embedding and add to cpc buffer
                # (last state might include useful task info)
                self.cpc_encoder.rollout_storage.insert(prev_state.clone(), action.clone(), next_state.clone(),
                                                        rew_raw.clone(),
                                                        done.clone(), task.clone(), None)

                # add the obs before reset to the policy storage
                self.policy_storage.next_state[step] = next_state.clone()

                # reset environments that are done
                done_indices = np.argwhere(done.cpu().flatten()).flatten()
                if len(done_indices) > 0:
                    next_state, task = utl.reset_env(self.envs, self.args,
                                                     indices=done_indices, state=next_state)

                # add experience to policy buffer
                self.policy_storage.insert(state=next_state, task=task if self.args.pass_task_to_policy else None,
                                           actions=action, rewards_raw=rew_raw, rewards_normalised=rew_normalised,
                                           value_preds=value, masks=masks_done, bad_masks=bad_masks, done=done,
                                           hidden_states=hidden_state.squeeze(0))

                prev_state = next_state
                self.frames += self.args.num_processes

            # --- UPDATE ---
            if self.args.precollect_len > self.frames:
                print(f'Precollect frames so far: {self.frames}', flush=True)
            else:
                # check if we are pre-training the representation learner
                if self.args.pretrain_len > self.iter_idx:
                    cpc_pretrain_stats = self.cpc_encoder.update_cpc()
                    if (self.iter_idx % 20) == 0:
                        print(f'Iteration {self.iter_idx}: {cpc_pretrain_stats.cpc_loss}',flush=True)
                # otherwise do the normal update (policy + vae)
                else:

                    train_policy_stats, train_representation_learner_stats, evaluation_loss = self.meta_update(state=prev_state,
                                                                                              task=task if self.args.pass_task_to_policy else None,
                                                                                              hidden=hidden_state.squeeze(
                                                                                                  0))

                    # log
                    run_stats = [action, self.policy_storage.action_log_probs, value]
                    with torch.no_grad():
                        self.log(run_stats, train_policy_stats, train_representation_learner_stats, evaluation_loss, start_time)

            # clean up after update
            self.policy_storage.after_update()

        self.envs.close()

    def encode_running_trajectory(self):
        """
        (Re-)Encodes (for each process) the entire current trajectory.
        Returns sample/mean/logvar and hidden state (if applicable) for the current timestep.
        :return:
        """

        # for each process, get the current batch (zero-padded obs/act/rew + length indicators)
        prev_obs, next_obs, act, rew, lens = self.cpc_encoder.rollout_storage.get_running_batch()

        # get embedding - will return (1+sequence_len) * batch * input_size -- includes the prior!
        all_hidden_states = self.cpc_encoder.encoder(actions=act,
                                                     states=next_obs,
                                                     rewards=rew,
                                                     hidden_state=None,
                                                     return_prior=True)

        # get the embedding / hidden state of the current time step (need to do this since we zero-padded)
        hidden_state = (torch.stack([all_hidden_states[lens[i]][i] for i in range(len(lens))])).to(device)

        return hidden_state

    def get_value(self, state, task, hidden_state):
        return self.policy.actor_critic.get_value(state=state, latent=hidden_state, task=task).detach()

    def meta_update(self, state, task, hidden):
        """
        Meta-update.
        Here the policy is updated for good average performance across tasks.
        :return:
        """
        evaluator_loss = None
        # update policy (if we are not pre-training, have enough data in the representation learner buffer, and are not at iteration 0)
        if self.iter_idx >= self.args.pretrain_len and self.iter_idx > 0:

            # bootstrap next value prediction
            with torch.no_grad():
                next_value = self.get_value(state=state, task=task, hidden_state=hidden)

            # compute returns for current rollouts
            self.policy_storage.compute_returns(next_value, self.args.policy_use_gae, self.args.policy_gamma,
                                                self.args.policy_tau,
                                                use_proper_time_limits=self.args.use_proper_time_limits)

            # update agent
            policy_train_stats = self.policy.policy_update(policy_storage=self.policy_storage)
            representation_learner_train_stats = self.cpc_encoder.update_cpc()
            if self.args.evaluate_representation and self.iter_idx >= self.args.evaluate_start_iter:
                evaluator_loss = self.cpc_encoder.update_latent_evaluator(
                    representation_learner_train_stats.tasks, representation_learner_train_stats.hidden_states)

        else:
            policy_train_stats = 0, 0, 0, 0, 0
            representation_learner_train_stats = None
            # pre-train the VAE
            if self.iter_idx < self.args.pretrain_len:
                self.vae.compute_vae_loss(update=True)

        return policy_train_stats, representation_learner_train_stats, evaluator_loss

    def log_evaluator_stats(self, log_prefix, stat):
        self.logger.add(f'cpc_evaluator/{log_prefix}_loss', stat, self.iter_idx)

    def log_cpc_stats(self, log_prefix, stats):
        self.logger.add(f'cpc/{log_prefix}_loss', stats.cpc_loss, self.iter_idx)
        self.logger.add(f'cpc/{log_prefix}_fraction_examples_reward_seen',
                        stats.fraction_examples_reward_seen,
                        self.iter_idx)
        self.logger.add(f'cpc/{log_prefix}_fraction_trajectories_reward_seen',
                        stats.fraction_trajectories_reward_seen, self.iter_idx)
        self.logger.add(f'cpc/{log_prefix}_avg_num_sampled', stats.num_sampled, self.iter_idx)

    def log(self, run_stats, train_policy_stats, train_cpc_stats, train_evaluation_loss, start_time):

        # --- visualise behaviour of policy ---

        if ((self.iter_idx + 1) % self.args.vis_interval == 0):
            ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None
            utl_eval.visualise_behaviour(args=self.args,
                                         policy=self.policy,
                                         image_folder=self.logger.full_output_folder,
                                         iter_idx=self.iter_idx,
                                         ret_rms=ret_rms,
                                         encoder=self.cpc_encoder,
                                         tasks=self.train_tasks,
                                         reward_decoder=self.cpc_encoder.reward_predictor if hasattr(self.cpc_encoder,
                                                                                                     'reward_predictor') else None,
                                         logger=self.logger
                                         )

        # --- evaluate policy ----
        if (self.iter_idx + 1) % self.args.eval_interval == 0:
            ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None
            returns_per_episode, eval_cpc_stats, eval_evaluator_stats = utl_eval.evaluate(args=self.args,
                                                                                          policy=self.policy,
                                                                                          ret_rms=ret_rms,
                                                                                          encoder=self.cpc_encoder,
                                                                                          iter_idx=self.iter_idx,
                                                                                          tasks=self.train_tasks,
                                                                                          )

            # log the return avg/std across tasks (=processes)
            returns_avg = returns_per_episode.mean(dim=0)
            returns_std = returns_per_episode.std(dim=0)
            delta_from_initial = (returns_per_episode[:, 1:].mean(dim=1) - returns_per_episode[:, 0]).mean()
            for k in range(len(returns_avg)):
                self.logger.add_hist('returns_per_iter_hist/episode_{}'.format(k + 1),
                                     returns_per_episode[:, k].reshape(-1), self.iter_idx)
                self.logger.add('return_avg_per_iter/episode_{}'.format(k + 1), returns_avg[k], self.iter_idx)
                self.logger.add('return_avg_per_frame/episode_{}'.format(k + 1), returns_avg[k], self.frames)
                self.logger.add('return_std_per_iter/episode_{}'.format(k + 1), returns_std[k], self.iter_idx)
                self.logger.add('return_std_per_frame/episode_{}'.format(k + 1), returns_std[k], self.frames)
            self.logger.add('delta_avg_from_intial', delta_from_initial, self.iter_idx)
            wandb.log({'return_avg_per_iter_episode_1': returns_avg[0]}, step=self.iter_idx)
            self.log_cpc_stats(stats=eval_cpc_stats, log_prefix='eval')
            if self.args.evaluate_representation and self.args.evaluate_start_iter < self.iter_idx:
                self.log_evaluator_stats('eval', eval_evaluator_stats)

            print(f"Updates {self.iter_idx}, "
                  f"Frames {self.frames}, "
                  f"FPS {int(self.frames / (time.time() - start_time))}, "
                  f"\n Mean return (eval): {returns_avg[-1].item()}"
                  f"\n Mean CPC loss (eval): {eval_cpc_stats.cpc_loss}"
                  )
            if self.args.evaluate_representation and self.args.evaluate_start_iter <= self.iter_idx:
                print(f"Updates {self.iter_idx}, evaluator loss: {eval_evaluator_stats.item()}")


        # --- save models ---

        if (self.iter_idx + 1) % self.args.save_interval == 0:
            save_path = os.path.join(self.logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            idx_labels = ['']
            if self.args.save_intermediate_models:
                idx_labels.append(int(self.iter_idx))

            for idx_label in idx_labels:

                torch.save(self.policy.actor_critic, os.path.join(save_path, f"policy{idx_label}.pt"))
                torch.save(self.cpc_encoder.encoder, os.path.join(save_path, f"encoder{idx_label}.pt"))
                if self.args.with_action_gru:
                    torch.save(self.cpc_encoder.action_gru, os.path.join(save_path, f"action_gru{idx_label}.pt"))
                torch.save(self.cpc_encoder.mlp, os.path.join(save_path, f"cpc_mlp{idx_label}.pt"))
                # save normalisation params of envs
                if self.args.norm_rew_for_policy:
                    rew_rms = self.envs.venv.ret_rms
                    utl.save_obj(rew_rms, save_path, f"env_rew_rms{idx_label}")
                # TODO: grab from policy and save?
                # if self.args.norm_obs_for_policy:
                #     obs_rms = self.envs.venv.obs_rms
                #     utl.save_obj(obs_rms, save_path, f"env_obs_rms{idx_label}")

        # --- log some other things ---
        if ((self.iter_idx + 2) % self.args.gradient_log_interval == 0):
            # utl.plot_grad_flow(self.cpc_encoder.encoder.named_parameters(), 'encoder_gradient_flow', self.logger, self.iter_idx)
            if self.args.with_action_gru:
                utl.plot_grad_flow(self.cpc_encoder.action_gru.named_parameters(), 'action_gru_gradient_flow', self.logger, self.iter_idx)

            #Uncomment to see log gradients for other components

            # utl.plot_grad_flow(self.cpc_encoder.mlp.named_parameters(), 'mlp_gradient_flow',self.logger, self.iter_idx)
            # utl.plot_grad_flow(self.policy.actor_critic.named_parameters(), 'policy_gradient_flow', self.logger, self.iter_idx)
            # log the average weights and gradients of all models (where applicable)
            models = [
                [self.policy.actor_critic, 'policy'],
                [self.cpc_encoder.encoder, 'encoder'],
                [self.cpc_encoder.mlp, 'cpc_mlp']]
            if self.args.with_action_gru:
                models += [[self.cpc_encoder.action_gru, 'action_gru']]
            for [model, name] in models:
                if model is not None:
                    param_list = list(model.parameters())
                    param_mean = np.mean([param_list[i].data.cpu().numpy().mean() for i in range(len(param_list))])
                    self.logger.add('weights/{}'.format(name), param_mean, self.iter_idx)
                    if name == 'policy':
                        self.logger.add('weights/policy_std', param_list[0].data.mean(), self.iter_idx)
                    if param_list[0].grad is not None:
                        param_grad_mean = []
                        for i in range(len(param_list)):
                            if param_list[i].grad is not None:
                                param_grad_mean.append(param_list[i].grad.cpu().numpy().mean())
                        param_grad_mean = np.array(param_grad_mean).mean()
                        self.logger.add('gradients/{}'.format(name), param_grad_mean, self.iter_idx)

        if ((self.iter_idx + 1) % self.args.log_interval == 0) and (train_policy_stats is not None):

            # POLICY

            self.logger.add('environment/state_max', self.policy_storage.prev_state.max(), self.iter_idx)
            self.logger.add('environment/state_min', self.policy_storage.prev_state.min(), self.iter_idx)

            self.logger.add('environment/rew_max', self.policy_storage.rewards_raw.max(), self.iter_idx)
            self.logger.add('environment/rew_min', self.policy_storage.rewards_raw.min(), self.iter_idx)
            self.logger.add('environment/rew_mean', self.policy_storage.rewards_raw.sum(axis=0).mean(), self.iter_idx)

            self.logger.add('policy_losses/value_loss', train_policy_stats[0], self.iter_idx)
            self.logger.add('policy_losses/action_loss', train_policy_stats[1], self.iter_idx)
            self.logger.add('policy_losses/dist_entropy', train_policy_stats[2], self.iter_idx)
            self.logger.add('policy_losses/sum', train_policy_stats[3], self.iter_idx)
            self.logger.add('policy/clip_frac', train_policy_stats[4], self.iter_idx)

            self.logger.add('policy/action', run_stats[0].abs().float().mean(), self.iter_idx)
            self.logger.add_hist('policy/action_hist', run_stats[0].abs().flatten(), self.iter_idx)
            if hasattr(self.policy.actor_critic, 'logstd'):
                self.logger.add('policy/action_logstd', self.policy.actor_critic.dist.logstd.mean(), self.iter_idx)
            self.logger.add('policy/action_logprob', run_stats[1].mean(), self.iter_idx)
            self.logger.add('policy/value', run_stats[2].mean(), self.iter_idx)

            # REPRESENTATION
            self.log_cpc_stats(stats=train_cpc_stats, log_prefix='train')
            if train_evaluation_loss is not None:
                self.log_evaluator_stats(stat=train_evaluation_loss, log_prefix='train')
