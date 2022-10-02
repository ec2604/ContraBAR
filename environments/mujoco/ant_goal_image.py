import random
import matplotlib.pyplot as plt
import numpy as np
import torch

from collections import deque
from environments.mujoco.ant import AntEnv
from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AntGoalEnvImage(AntEnv):
    def __init__(self, max_episode_steps=200):  # , stochastic_moves=True):
        self.set_task(self.sample_tasks(1))
        self._max_episode_steps = max_episode_steps
        self.task_dim = 2
        self._frames = deque([], maxlen=3)
        # self.wind = np.array([0, 0])
        # self.wind = np.array([random.random() * 0.1 - 0.05,random.random() * 0.1 - 0.05])
        super(AntGoalEnvImage, self).__init__()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        # qpos = self.sim.data.qpos
        # qvel = self.sim.data.qvel
        # qpos[:2] += self.wind
        # self.set_state(qpos, qvel)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal_pos))  # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        # reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        reward = goal_reward - ctrl_cost
        state = self.state_vector()
        done = False
        curr_obs = self._get_obs()
        self._frames.append(curr_obs)
        if len(self._frames) != 3:
            self.reset(again=False)
        ob = self.get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            task=self.get_task(),
            state=state
        )

    def viewer_setup(self):
        #     self.viewer.cam.trackbodyid = 0  # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 0.5  # how much you "zoom in", model.stat.extent is the max limits of the arena

    #     self.viewer.cam.lookat[0] += 0.5  # x,y,z offset from the object (works if trackbodyid=-1)
    #     self.viewer.cam.lookat[1] += 0.5
    #     self.viewer.cam.lookat[2] += 0.5
    #     self.viewer.cam.elevation = -90  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
    #     self.viewer.cam.azimuth = 90  # camera rotation around the camera's vertical axis
    def sample_tasks(self, num_tasks):
       # a = np.array([random.random() for _ in range(num_tasks)]) * np.pi
        a = np.array([0.5*np.pi])
        # r = 1 * np.array([random.random() for _ in range(num_tasks)]) ** 0.5
        # a = np.array([0.75 for _ in range(num_tasks)]) * np.pi
        r = 1
        return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

    def set_task(self, task):
        self.goal_pos = task

    def get_task(self):
        return np.array(self.goal_pos)

    def _get_obs(self):
        return utl.rgb2gray(self.render('rgb_array', height=64, width=64))

    def get_obs(self):
        assert len(self._frames) == 3
        return np.concatenate(list(self._frames), axis=0).astype(np.uint8)

    def reset(self, again=True):
        """
        Reset the environment. This should *NOT* reset the task!
        Resetting the task is handled in the contrabar wrapper (see wrappers.py).
        """
        if again:
            super().reset()
        obs = self._get_obs()
        for _ in range(3):
            self._frames.append(obs)
        return self.get_obs()

    def reset_task(self, task):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)
        self.reset()

    def visualise_behaviour(self, env,
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
        env.reset_task()
        state, task = utl.reset_env(env, args)
        start_obs_raw = state.clone()
        task = task.view(-1) if task is not None else None

        # initialise actions and rewards (used as initial input to policy if we have a recurrent policy)
        if hasattr(args, 'hidden_size'):
            hidden_state = torch.zeros((1, args.hidden_size)).to(device)
        else:
            hidden_state = None

        # keep track of what task we're in and the position of the cheetah
        pos = [[] for _ in range(args.max_rollouts_per_task)]
        start_pos = unwrapped_env.get_body_com("torso")[:2].copy()

        for episode_idx in range(num_episodes):

            curr_rollout_rew = []
            pos[episode_idx].append(start_pos)

            if episode_idx == 0:
                if encoder is not None:
                    # reset to prior
                    current_hidden_state = encoder.prior(1)
                    current_hidden_state.to(device)
            episode_hidden_states[episode_idx].append(current_hidden_state[0].clone())

            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    episode_prev_obs[episode_idx].append(start_obs_raw.clone())
                else:
                    episode_prev_obs[episode_idx].append(state.clone())
                # act
                _, action = utl.select_action_cpc(args=args, policy=policy, deterministic=True,
                                                  hidden_latent=current_hidden_state.squeeze(0), state=state, task=task)

                (state, task), (rew, rew_normalised), done, info = utl.env_step(env, action, args)
                state = state.float().to(device)
                task = task.view(-1) if task is not None else None

                # keep track of position
                pos[episode_idx].append(unwrapped_env.get_body_com("torso")[:2].copy())

                if encoder is not None:
                    # update task embedding
                    current_hidden_state = encoder(
                        action.reshape(1, -1).float().to(device), state, rew.reshape(1, -1).float().to(device),
                        current_hidden_state, return_prior=False)

                    episode_hidden_states[episode_idx].append(current_hidden_state[0].clone())

                episode_next_obs[episode_idx].append(state.clone())
                episode_rewards[episode_idx].append(rew.clone())
                episode_actions[episode_idx].append(action.clone())

                if info[0]['done_mdp'] and not done:
                    start_obs_raw = info[0]['start_state']
                    start_obs_raw = torch.from_numpy(start_obs_raw).float().to(device).unsqueeze(0)
                    start_pos = unwrapped_env.get_body_com("torso")[:2].copy()
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)

        # clean up

        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.stack(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]
        # kwargs['logger'].add_video('behaviour_video_rollout_1',
        #                            episode_prev_obs[0][:, 2, ...].unsqueeze(0).unsqueeze(2).to(torch.uint8), iter_idx)
        # kwargs['logger'].add_video('behaviour_video_rollout_2',
        #                            episode_prev_obs[1][:, 2, ...].unsqueeze(0).unsqueeze(2).to(torch.uint8), iter_idx)
        # plot the movement of the ant
        # print(pos)
        # figure = plt.figure(figsize=(5, 4 * num_episodes))
        min_dim = -3.5
        max_dim = 3.5
        span = max_dim - min_dim

        for i in range(num_episodes):
            figure = plt.figure()
            x = list(map(lambda p: p[0], pos[i]))
            y = list(map(lambda p: p[1], pos[i]))
            angle = np.linspace(0, np.pi)
            goal_range_x = np.cos(angle)
            goal_range_y = np.sin(angle)
            plt.plot(goal_range_x, goal_range_y, 'k--', alpha=0.1)
            plt.plot(x[0], y[0], 'bo')

            plt.plot(x, y, '-')

            curr_task = env.get_task()[0]
            plt.title(f'task: ({curr_task[0]:.2f},{curr_task[1]:.2f}) , Cumulative reward: {episode_rewards[i].sum():.2f}')
            if 'Goal' in args.env_name:
                plt.plot(curr_task[0], curr_task[1], 'rx')
            if hasattr(self, 'goal_radius'):
                circle1 = plt.Circle(curr_task, self.goal_radius, color='c', alpha=0.2, edgecolor='none')
                plt.gca().add_artist(circle1)
            plt.ylabel('y-position (ep {})'.format(i), fontsize=15)

            if i == num_episodes - 1:
                plt.xlabel('x-position', fontsize=15)
                plt.ylabel('y-position (ep {})'.format(i), fontsize=15)
            plt.xlim(-1.2, 1.2)
            plt.ylim(-0.5, 1.2)

            plt.tight_layout()
            kwargs['logger'].add_figure(f'belief_{i}', figure, iter_idx)
            # if image_folder is not None:
            #     plt.savefig('{}/{}_ep_{}_behaviour'.format(image_folder, iter_idx, i))
            #     plt.close()
            # else:
            #     plt.show()

        if not return_pos:
            return episode_hidden_states, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns
        else:
            return episode_hidden_states, \
                   episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns, pos


class SparseAntGoalEnvImage(AntGoalEnvImage):
    def __init__(self, max_episode_steps=200, goal_radius=0.2):  # , stochastic_moves=True):
        self.goal_radius = goal_radius
        # self.wind = np.array([0, 0])
        # self.wind = np.array([random.random() * 0.1 - 0.05,random.random() * 0.1 - 0.05])
        super().__init__(max_episode_steps)

    def step(self, action):
        ob, reward, done, d = super().step(action)
        if self.is_goal_state():
            ob[2, -20:, ...] = 0
        sparse_reward = self.sparsify_rewards(d)
        return ob, sparse_reward, done, d

    def render_(self, mode='rgb'):
        ob = self.get_obs()
        if self.is_goal_state():
            ob[2, -20:, ...] = 0
        return ob

    def is_goal_state(self, state=None):
        if state is None:
            state = np.array(self.get_body_com("torso"))
        if np.linalg.norm(state[:2] - self.goal_pos) <= self.goal_radius:
            return True
        else:
            return False

    def sparsify_rewards(self, d):
        non_goal_reward_keys = []
        for key in d.keys():
            if key.startswith('reward') and key != "reward_goal":
                non_goal_reward_keys.append(key)
        non_goal_rewards = np.sum([d[reward_key] for reward_key in non_goal_reward_keys])
        #non_goal_rewards = d['reward_ctrl']
        sparse_goal_reward = 1. if self.is_goal_state() else 0.
        return non_goal_rewards + sparse_goal_reward


class OfflineAntGoalEnvImage(AntEnv):
    def __init__(self, max_episode_steps=200):  # , stochastic_moves=True):
        self.set_task(self.sample_tasks(1))
        self._max_episode_steps = max_episode_steps
        self.task_dim = 2
        self._frames = deque([], maxlen=3)
        # self.wind = np.array([0, 0])
        # self.wind = np.array([random.random() * 0.1 - 0.05,random.random() * 0.1 - 0.05])
        super(OfflineAntGoalEnvImage, self).__init__()


    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        # qpos = self.sim.data.qpos
        # qvel = self.sim.data.qvel
        # qpos[:2] += self.wind
        # self.set_state(qpos, qvel)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal_pos))  # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        # reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        reward = goal_reward - ctrl_cost - contact_cost
        state_vec = self.state_vector()
        done = False
        curr_obs = self._get_obs()
        self._frames.append(curr_obs)
        if len(self._frames) != 3:
            self.reset(again=False)
        ob = self.get_obs()
        state = self.get_state()
        return state, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            task=self.get_task(),
            state=state
        )

    def viewer_setup(self):
        #     self.viewer.cam.trackbodyid = 0  # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 0.5  # how much you "zoom in", model.stat.extent is the max limits of the arena

    #     self.viewer.cam.lookat[0] += 0.5  # x,y,z offset from the object (works if trackbodyid=-1)
    #     self.viewer.cam.lookat[1] += 0.5
    #     self.viewer.cam.lookat[2] += 0.5
    #     self.viewer.cam.elevation = -90  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
    #     self.viewer.cam.azimuth = 90  # camera rotation around the camera's vertical axis
    def sample_tasks(self, num_tasks):
        a = np.array([random.random() for _ in range(num_tasks)]) * np.pi
        # r = 1 * np.array([random.random() for _ in range(num_tasks)]) ** 0.5
        # a = np.array([0.75 for _ in range(num_tasks)]) * np.pi
        r = 1
        return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

    def set_task(self, task):
        self.goal_pos = task

    def get_task(self):
        return np.array(self.goal_pos)

    def _get_obs(self):
        return utl.rgb2gray(self.render('rgb_array', height=64, width=64))

    def get_obs(self):
        assert len(self._frames) == 3
        return np.concatenate(list(self._frames), axis=0)

    def get_state(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset(self, again=True):
        """
        Reset the environment. This should *NOT* reset the task!
        Resetting the task is handled in the contrabar wrapper (see wrappers.py).
        """
        if again:
            super().reset()
        obs = self._get_obs()
        for _ in range(3):
            self._frames.append(obs)
        return self.get_state()#self.get_obs()

    def reset_task(self, task):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)
        self.reset()

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
        env.reset_task()
        state, task = utl.reset_env(env, args)
        start_obs_raw = state.clone()
        task = task.view(-1) if task is not None else None

        # initialise actions and rewards (used as initial input to policy if we have a recurrent policy)
        if hasattr(args, 'hidden_size'):
            hidden_state = torch.zeros((1, args.hidden_size)).to(device)
        else:
            hidden_state = None

        # keep track of what task we're in and the position of the cheetah
        pos = [[] for _ in range(args.max_rollouts_per_task)]
        start_pos = unwrapped_env.get_body_com("torso")[:2].copy()

        for episode_idx in range(num_episodes):

            curr_rollout_rew = []
            pos[episode_idx].append(start_pos)

            if episode_idx == 0:
                if encoder is not None:
                    # reset to prior
                    current_hidden_state = encoder.prior(1)
                    current_hidden_state.to(device)
            episode_hidden_states[episode_idx].append(current_hidden_state[0].clone())

            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    episode_prev_obs[episode_idx].append(start_obs_raw.clone())
                else:
                    episode_prev_obs[episode_idx].append(state.clone())
                # act
                _, action = utl.select_action_cpc(args=args, policy=policy, deterministic=True,
                                                  hidden_latent=current_hidden_state.squeeze(0), state=state, task=task)

                (state, task), (rew, rew_normalised), done, info = utl.env_step(env, action, args)
                state = state.float().to(device)
                underlying_state = torch.from_numpy(unwrapped_env.render_()).unsqueeze(0).to(device)
                if done[0]:
                    underlying_state = torch.cat([underlying_state, torch.ones(underlying_state.shape[0],1,*underlying_state.shape[2:]).to(device)], dim=1).float()
                else:
                    underlying_state = torch.cat([underlying_state, torch.zeros(underlying_state.shape[0],1,*underlying_state.shape[2:]).to(device)],
                                                  dim=1).float()
                task = task.view(-1) if task is not None else None

                # keep track of position
                pos[episode_idx].append(unwrapped_env.get_body_com("torso")[:2].copy())

                if encoder is not None:
                    # update task embedding
                    current_hidden_state = encoder(
                        action.reshape(1, -1).float().to(device), underlying_state, rew.reshape(1, -1).float().to(device),
                        current_hidden_state, return_prior=False)

                    episode_hidden_states[episode_idx].append(current_hidden_state[0].clone())

                episode_next_obs[episode_idx].append(state.clone())
                episode_rewards[episode_idx].append(rew.clone())
                episode_actions[episode_idx].append(action.clone())

                if info[0]['done_mdp'] and not done:
                    start_obs_raw = info[0]['start_state']
                    start_obs_raw = torch.from_numpy(start_obs_raw).float().to(device).unsqueeze(0)
                    start_pos = unwrapped_env.get_body_com("torso")[:2].copy()
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)

        # clean up

        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.stack(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]
        # kwargs['logger'].add_video('behaviour_video_rollout_1',
        #                            episode_prev_obs[0][:, 2, ...].unsqueeze(0).unsqueeze(2).to(torch.uint8), iter_idx)
        # kwargs['logger'].add_video('behaviour_video_rollout_2',
        #                            episode_prev_obs[1][:, 2, ...].unsqueeze(0).unsqueeze(2).to(torch.uint8), iter_idx)
        # plot the movement of the ant
        # print(pos)
        figure = plt.figure(figsize=(5, 4 * num_episodes))
        min_dim = -3.5
        max_dim = 3.5
        span = max_dim - min_dim

        for i in range(num_episodes):
            figure = plt.figure()

            x = list(map(lambda p: p[0], pos[i]))
            y = list(map(lambda p: p[1], pos[i]))
            plt.plot(x[0], y[0], 'bo')

            plt.scatter(x, y, 1, 'g')

            curr_task = env.get_task()[0]
            plt.title('task: {}'.format(curr_task), fontsize=15)
            if 'Goal' in args.env_name:
                plt.plot(curr_task[0], curr_task[1], 'rx')

            plt.ylabel('y-position (ep {})'.format(i), fontsize=15)

            if i == num_episodes - 1:
                plt.xlabel('x-position', fontsize=15)
                plt.ylabel('y-position (ep {})'.format(i), fontsize=15)
            plt.xlim(min_dim - 0.05 * span, max_dim + 0.05 * span)
            plt.ylim(min_dim - 0.05 * span, max_dim + 0.05 * span)

        plt.tight_layout()
        kwargs['logger'].add_figure('belief', figure, iter_idx)
        if image_folder is not None:
            plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
            plt.close()
        else:
            plt.show()

        if not return_pos:
            return episode_hidden_states, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns
        else:
            return episode_hidden_states, \
                   episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns, pos

class OfflineSparseAntGoalEnvImage(OfflineAntGoalEnvImage):
    def __init__(self, max_episode_steps=200, goal_radius=0.3):  # , stochastic_moves=True):
        self.goal_radius = goal_radius
        # self.wind = np.array([0, 0])
        # self.wind = np.array([random.random() * 0.1 - 0.05,random.random() * 0.1 - 0.05])
        super().__init__(max_episode_steps)

    def render_(self, mode='rgb'):
        ob = self.get_obs()
        if self.is_goal_state():
            ob[2, -20:, ...] = 0
        return ob

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(d)
        return ob, sparse_reward, done, d

    def is_goal_state(self, state=None):
        if state is None:
            state = np.array(self.get_body_com("torso"))
        if np.linalg.norm(state[:2] - self.goal_pos) <= self.goal_radius:
            return True
        else:
            return False

    def sparsify_rewards(self, d):
        non_goal_reward_keys = []
        for key in d.keys():
            if key.startswith('reward') and key != "reward_goal":
                non_goal_reward_keys.append(key)
        non_goal_rewards = np.sum([d[reward_key] for reward_key in non_goal_reward_keys])
        #non_goal_rewards = d['reward_ctrl']
        sparse_goal_reward = 1. if self.is_goal_state() else 0.
        return non_goal_rewards + sparse_goal_reward