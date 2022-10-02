import gym
import dmc2gym
import numpy as np
import torch

from collections import deque
import matplotlib.pyplot as plt
import random
from utils import helpers as utl
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReacherEnv(gym.Env):
    def __init__(self, action_repeat, max_episode_steps=200, n_tasks=2, dense=False, **kwargs):
        super(ReacherEnv, self).__init__()
        self.env = dmc2gym.make(
            domain_name='reacher',
            task_name='easy',
            seed=1,
            visualize_reward=False,
            from_pixels=False,
            # height=image_size,
            # width=image_size,
            frame_skip=action_repeat
        )

        self.dense = dense
        self.num_frames = 3
        # self.observation_space = self.env.observation_space -2
        #shp = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            #shape=((shp[0] * self.num_frames,) + shp[1:]),
            shape=(self.env.observation_space.shape[0]-2,),
            dtype=self.env.observation_space.dtype
        )
        self._frames = deque([], maxlen=self.num_frames)
        self.env._max_episode_steps = max_episode_steps
        self.max_episode_steps = max_episode_steps
        self.action_space = self.env.action_space
        self.task_object = self.env.unwrapped._env.task
        self.goal_radius = 0.05
        self.task_dim = 2
        self.task_object._target_size = self.goal_radius
        self.goals = [np.array([np.random.uniform(np.pi / 2, np.pi), 0.24]) for _ in range(n_tasks)]
        #np.random.uniform(.1, .13)
        #self.goals = [np.array([np.random.uniform(0, np.pi / 2), np.random.uniform(.1, .13)]) for _ in range(n_tasks)]
        #self.goals = [np.array([2.226, 0.122])]
        self.tasks = self.goals
        self.reset_task()

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_task(self):
        """
        Return a task description, such as goal position or target velocity.
        """
        return self._goal

    def set_goal(self, goal, pos=None):
        """
        Sets goal manually. Mainly used for reward relabelling.
        """
        self._goal = np.asarray(goal)
        self.task_object.angle = goal[0]
        self.task_object.radius = goal[1]
        range_min, range_max = self.env.physics.model.jnt_range[1]
        if pos is None:
            self.task_object.wrist_pos = range_min + 0.5*(range_max-range_min)
            self.task_object.shoulder_pos = np.pi / 2# + random.random()*(2*np.pi)
        else:
            self.task_object.wrist_pos = range_min
            self.task_object.shoulder_pos = pos
        return self.reset()

    def reset_task(self, idx=None, **kwargs):
        """
        Reset the task, either at random (if idx=None) or the given task.
        """
        if idx is None:
            if 'task' in kwargs:
                self.set_goal(kwargs['task'])
            else:
                r = random.random()*2 -1
                self._goal = np.array([np.pi / 4, r*0.24])#self.goals[int(np.random.randint(0,len(self.goals),1))]
                self.set_goal(self._goal)
        else:
            self._goal = self.goals[idx]

        self.reset()

    def step(self, action):
        """
        Execute one step in the environment.
        Should return: state, reward, done, info
        where info has to include a field 'task'.
        """
        obs, reward, done, info = self.env.step(action)
        # obs_gray = rgb2gray(obs)
        # if not self.dense:
        #     if self.reward(None, None) > 0:
        #         obs_gray[:20,:20] = 0
        # self._frames.append(obs_gray)
        # self._frames.append(obs)

        info['state'] = self.task_object.get_state(self.env.physics)
        if self.dense:
            info['reward_dense'] = self.reward(None, None)
            info['reward_sparse'] = self.task_object.get_sparse_reward(self.env.physics)
        # return self._get_obs(), self.reward(None, None), done, info
        obs_fixed = np.concatenate([obs[:2], obs[4:]])
        return obs_fixed, self.reward(None, None), done, info


    def reward(self, state, action):
        """
        Computes reward function of task.
        Returns the reward
        """
        if self.dense:
            return self.task_object.get_reward(self.env.physics)
        else:
            return self.task_object.get_sparse_reward(self.env.physics)

    def reset(self):
        """
        Reset the environment. This should *NOT* reset the task!
        Resetting the task is handled in the contrabar wrapper (see wrappers.py).
        """
        obs = self.env.reset()
        # for _ in range(self.num_frames):
        #     self._frames.append(rgb2gray(obs))
        #     self._frames.append(obs)
        obs_fixed = np.concatenate([obs[:2], obs[4:]])
        return obs_fixed

    def seed(self, seed=None):
        super(ReacherEnv, self).seed(seed)
        return self.env.seed(seed)

    def is_goal_state(self):
        if self.task_object.get_sparse_reward(self.env.physics) == 1:
            return True
        else:
            return False

    # def _get_obs(self):
    #     assert len(self._frames) == self.num_frames
    #     return np.concatenate(list(self._frames), axis=0)

    def get_state(self):
        return self.task_object.get_state(self.env.physics)

    def get_all_task_idx(self):
        return range(len(self.goals))

    def plot_env(self):
        ax = plt.gca()
        # fix visualization
        plt.axis('scaled')
        # ax.set_xlim(-1.25, 1.25)
        ax.set_xlim(-0.32, 0.32)
        # ax.set_ylim(-0.25, 1.25)
        ax.set_ylim(-0.32, 0.32)
        ax.axhline(y=0, c='grey', ls='--')
        ax.axvline(x=0, c='grey', ls='--')
        plt.xticks([])
        plt.yticks([])
        goal_x = self._goal[1]*np.sin(self._goal[0])
        goal_y = self._goal[1]*np.cos(self._goal[0])
        circle = plt.Circle((goal_x, goal_y),
                            radius=0.01+self.goal_radius if hasattr(self, 'goal_radius') else 0.1,
                            alpha=0.3)
        circle_general = plt.Circle((0, 0),
                            radius=0.2,
                            alpha=0.3)
        ax.add_artist(circle)
        ax.add_artist(circle_general)
        return ax

    def plot_behavior(self, states, plot_env=True, **kwargs):
        ax = self.plot_env()
        ax.plot(states[1:, 0], states[1:, 1], **kwargs)

    def reward_from_state(self, state, action):
        if self.dense:
            return self.task_object.get_reward_from_state_dense(self.env.physics, state)
        else:
            return self.task_object.get_reward_from_state_sparse(self.env.physics, state)

    def print_stuff(self):
        self.task_object.print_stuff(self.env.physics)

    @staticmethod
    def visualise_behaviour(env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            image_folder=None,
                            return_pos=False,
                            **kwargs,
                            ):

        num_episodes = args.max_rollouts_per_task
        unwrapped_env = env.venv.unwrapped.envs[0].unwrapped

        # --- initialise things we want to keep track of ---

        episode_prev_state = [[] for _ in range(num_episodes)]
        episode_next_state = [[] for _ in range(num_episodes)]
        episode_prev_obs = [[] for _ in range(num_episodes)]
        episode_next_obs = [[] for _ in range(num_episodes)]
        episode_actions = [[] for _ in range(num_episodes)]
        episode_rewards = [[] for _ in range(num_episodes)]

        episode_returns = []
        episode_lengths = []

        if encoder is not None:
            episode_hidden_states = [[] for _ in range(num_episodes)]
        else:
            episode_hidden_states = None


        # --- roll out policy ---

        # (re)set environment


        obs, task = utl.reset_env(env, args)
        start_obs = obs.clone()
        start_state = env.venv.envs[0].get_state().copy()


        for episode_idx in range(num_episodes):

            curr_rollout_rew = []

            if episode_idx == 0:
                if encoder is not None:
                    # reset to prior
                    current_hidden_state = encoder.prior(1)
                    current_hidden_state.to(device)
            episode_hidden_states[episode_idx].append(current_hidden_state[0].clone())

            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    episode_prev_obs[episode_idx].append(start_obs.clone())
                    episode_prev_state[episode_idx].append(start_state.copy())
                else:
                    episode_prev_obs[episode_idx].append(obs.clone())
                    episode_prev_state[episode_idx].append(state.copy())
                # act
                _, action = utl.select_action_cpc(args=args, policy=policy, deterministic=True,
                                                  hidden_latent=current_hidden_state.squeeze(0), state=obs, task=task)
                (obs, task), (rew, rew_normalised), done, info = utl.env_step(env, action, args)
                state = env.venv.envs[0].get_state()
                #obs = obs.reshape((1, -1)).float().to(device)


                if encoder is not None:
                    # update task embedding
                    current_hidden_state = encoder(
                        action.reshape(1, -1).float().to(device), obs, rew.reshape(1, -1).float().to(device),
                        current_hidden_state, return_prior=False)

                    episode_hidden_states[episode_idx].append(current_hidden_state[0].clone())

                episode_next_obs[episode_idx].append(obs.clone())
                episode_next_state[episode_idx].append(state.copy())
                episode_rewards[episode_idx].append(rew.clone())
                episode_actions[episode_idx].append(action.reshape(1, -1).clone())

                if info[0]['done_mdp'] and not done:
                    start_obs = info[0]['start_state']
                    start_obs = torch.from_numpy(start_obs).float().to(device).unsqueeze(0)
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)

        #clean up
        episode_prev_state = [np.vstack(episode) for episode in episode_prev_state]
        episode_next_state = [np.vstack(episode) for episode in episode_next_state]
        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.cat(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]
        # kwargs['logger'].add_video('behaviour_video_ep_0', episode_prev_obs[0][:, 2, ...].unsqueeze(0).unsqueeze(2).to(torch.uint8), iter_idx)
        # kwargs['logger'].add_video('behaviour_video_ep_1', episode_prev_obs[1][:, 2, ...].unsqueeze(0).unsqueeze(2).to(torch.uint8), iter_idx)

        for j in range(num_episodes):
            # plot the movement of the reacher
            figure = plt.figure()
            # fix visualization
            plt.axis('scaled')
            plt.xlim([-0.32, 0.32])
            # ax.set_ylim(-0.25, 1.25)
            plt.ylim([-0.32, 0.32])
            plt.axhline(y=0, c='grey', ls='--')
            plt.axvline(x=0, c='grey', ls='--')
            plt.xticks([])
            plt.yticks([])
            #for some reason goals are flipped
            goal_y = unwrapped_env._goal[1] * np.sin(unwrapped_env._goal[0])
            goal_x = unwrapped_env._goal[1] * np.cos(unwrapped_env._goal[0])
            circle = plt.Circle((goal_y, goal_x),
                                radius=0.01 + unwrapped_env.goal_radius if hasattr(unwrapped_env, 'goal_radius') else 0.1,
                                alpha=0.3)
            #angle = np.linspace(np.pi / 2, np.pi /2 + np.pi, 100)
            angle = np.pi / 4
            #goal_range = 0.24 * np.array((np.cos(angle), np.sin(angle)))
            r = np.linspace(-0.24, 0.24, 100)
            goal_range_x = np.cos(angle) * r
            goal_range_y = np.sin(angle) * r
            #plt.plot(goal_range[1], goal_range[0], 'k--', alpha=0.1)
            plt.plot(goal_range_y, goal_range_x, 'k--', alpha=0.1)
            plt.gca().add_artist(circle)
            plt.scatter(goal_y, goal_x, marker='x', color='k', s=50)
            plt.plot(episode_prev_state[j][:, 0], episode_prev_state[j][:, 1])
            plt.scatter(episode_prev_state[j][0,0], episode_prev_state[j][0, 1], color='r',marker='o')
            plt.quiver(episode_prev_state[j][:-1, 0], episode_prev_state[j][:-1, 1],
                       episode_prev_state[j][1:, 0] - episode_prev_state[j][:-1, 0],
                       episode_prev_state[j][1:, 1] - episode_prev_state[j][:-1, 1], scale_units='xy', angles='xy', scale=2,
                       color='red')
            #Classification
            # if j == 1 and kwargs['belief_evaluator'] is not None:
            #     r = 0.24
            #     num_points_semicircle = 50
            #     angles = np.linspace(np.pi / 2, np.pi / 2 + np.pi, num=num_points_semicircle)
            #     x, y = r * np.cos(angles), r * np.sin(angles)
            #     ## Does the hidden state matter?
            #     belief = 1. - torch.sigmoid(kwargs['belief_evaluator'](episode_hidden_states[0][-1])).detach().cpu().numpy().flatten()
            #     plt.scatter(y,x, c= belief, cmap='gray')
            #Regression
            # if j == 1 and kwargs['belief_evaluator'] is not None:
            #     x, y = kwargs['belief_evaluator'](episode_hidden_states[j][0]).detach().cpu().numpy().flatten()
            #     plt.scatter(y,x, s=20, cmap='gray')

            plt.title(f'Reward: {episode_rewards[j].sum()}, max reward: {episode_rewards[j].max()}')
            kwargs['logger'].add_figure(f'belief_{j}', figure, iter_idx)
            if image_folder is not None:
                plt.savefig('{}/{}_ep_{}_behaviour'.format(image_folder, iter_idx, j))
                plt.close()
            else:
                plt.show()

        return episode_hidden_states, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
               episode_returns

    @staticmethod
    def visualise_behaviour_for_script(env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            image_folder=None,
                            return_pos=False,
                            **kwargs,
                            ):

        num_episodes = args.max_rollouts_per_task
        unwrapped_env = env.venv.unwrapped.envs[0].unwrapped

        # --- initialise things we want to keep track of ---

        episode_prev_state = [[] for _ in range(num_episodes)]
        episode_next_state = [[] for _ in range(num_episodes)]
        episode_prev_obs = [[] for _ in range(num_episodes)]
        episode_next_obs = [[] for _ in range(num_episodes)]
        episode_actions = [[] for _ in range(num_episodes)]
        episode_rewards = [[] for _ in range(num_episodes)]

        episode_returns = []
        episode_lengths = []

        if encoder is not None:
            episode_hidden_states = [[] for _ in range(num_episodes)]
        else:
            episode_hidden_states = None


        # --- roll out policy ---

        # (re)set environment
        goal_pos = [np.array([np.pi/2, 0.2])]
        obs, task = utl.reset_env(env, args, task=goal_pos)
        start_obs = obs.clone()
        start_state = env.venv.envs[0].get_state().copy()


        for episode_idx in range(num_episodes):

            curr_rollout_rew = []

            if episode_idx == 0:
                if encoder is not None:
                    # reset to prior
                    current_hidden_state = encoder.prior(1)
                    current_hidden_state.to(device)
            episode_hidden_states[episode_idx].append(current_hidden_state[0].clone())

            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    episode_prev_obs[episode_idx].append(start_obs.clone())
                    episode_prev_state[episode_idx].append(start_state.copy())
                else:
                    episode_prev_obs[episode_idx].append(obs.clone())
                    episode_prev_state[episode_idx].append(state.copy())
                # act
                _, action = utl.select_action_cpc(args=args, policy=policy, deterministic=True,
                                                  hidden_latent=current_hidden_state.squeeze(0), state=obs, task=task)
                (obs, task), (rew, rew_normalised), done, info = utl.env_step(env, action, args)
                state = env.venv.envs[0].get_state()
                #obs = obs.reshape((1, -1)).float().to(device)


                if encoder is not None:
                    # update task embedding
                    current_hidden_state = encoder(
                        action.reshape(1, -1).float().to(device), obs, rew.reshape(1, -1).float().to(device),
                        current_hidden_state, return_prior=False)

                    episode_hidden_states[episode_idx].append(current_hidden_state[0].clone())

                episode_next_obs[episode_idx].append(obs.clone())
                episode_next_state[episode_idx].append(state.copy())
                episode_rewards[episode_idx].append(rew.clone())
                episode_actions[episode_idx].append(action.reshape(1, -1).clone())

                if info[0]['done_mdp'] and not done:
                    start_obs = info[0]['start_state']
                    start_obs = torch.from_numpy(start_obs).float().to(device).unsqueeze(0)
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)

        #clean up
        episode_prev_state = [np.vstack(episode) for episode in episode_prev_state]
        episode_next_state = [np.vstack(episode) for episode in episode_next_state]
        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.cat(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]
        # kwargs['logger'].add_video('behaviour_video_ep_0', episode_prev_obs[0][:, 2, ...].unsqueeze(0).unsqueeze(2).to(torch.uint8), iter_idx)
        # kwargs['logger'].add_video('behaviour_video_ep_1', episode_prev_obs[1][:, 2, ...].unsqueeze(0).unsqueeze(2).to(torch.uint8), iter_idx)

        for j in range(num_episodes):
            # plot the movement of the reacher
            figure = plt.figure()
            # fix visualization
            plt.axis('scaled')
            plt.xlim([-0.32, 0.32])
            # ax.set_ylim(-0.25, 1.25)
            plt.ylim([-0.32, 0.32])
            plt.axhline(y=0, c='grey', ls='--')
            plt.axvline(x=0, c='grey', ls='--')
            plt.xticks([])
            plt.yticks([])
            #for some reason goals are flipped
            goal_y = unwrapped_env._goal[1] * np.sin(unwrapped_env._goal[0])
            goal_x = unwrapped_env._goal[1] * np.cos(unwrapped_env._goal[0])
            circle = plt.Circle((goal_y, goal_x),
                                radius=0.01 + unwrapped_env.goal_radius if hasattr(unwrapped_env, 'goal_radius') else 0.1,
                                alpha=0.3)
            angle = np.linspace(np.pi / 2, np.pi /2 + np.pi, 100)
            goal_range = 0.24 * np.array((np.cos(angle), np.sin(angle)))
            plt.plot(goal_range[1], goal_range[0], 'k--', alpha=0.1)
            plt.gca().add_artist(circle)
            plt.plot(episode_prev_state[j][:, 0], episode_prev_state[j][:, 1])
            plt.scatter(episode_prev_state[j][0,0], episode_prev_state[j][0, 1], color='r',marker='o')
            plt.quiver(episode_prev_state[j][:-1, 0], episode_prev_state[j][:-1, 1],
                       episode_prev_state[j][1:, 0] - episode_prev_state[j][:-1, 0],
                       episode_prev_state[j][1:, 1] - episode_prev_state[j][:-1, 1], scale_units='xy', angles='xy', scale=2,
                       color='red')
            #Classification
            # if j == 1 and kwargs['belief_evaluator'] is not None:
            #     r = 0.2
            #     num_points_semicircle = 150
            #     angles = np.linspace(np.pi / 2, np.pi / 2 + np.pi, num=num_points_semicircle)
            #     x, y = r * np.cos(angles), r * np.sin(angles)
            #     belief = 1. - torch.sigmoid(kwargs['belief_evaluator'](episode_hidden_states[j][0])).detach().cpu().numpy().flatten()
            #     plt.scatter(y,x, c= belief, cmap='gray')
            #Regression
            # if j == 1 and kwargs['belief_evaluator'] is not None:
            #     x, y = kwargs['belief_evaluator'](episode_hidden_states[j][0]).detach().cpu().numpy().flatten()
            #     plt.scatter(y,x, s=20, cmap='gray')

            plt.title(f'Reward: {episode_rewards[j].sum()}, max reward: {episode_rewards[j].max()}')
            # kwargs['logger'].add_figure(f'belief_{j}', figure, iter_idx)
            if image_folder is not None:
                plt.savefig('{}/{}_ep_{}_behaviour'.format(image_folder, iter_idx, j))
                plt.close()
            else:
                plt.show()

        return episode_hidden_states, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
               episode_returns