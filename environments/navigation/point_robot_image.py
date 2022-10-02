import matplotlib as mpl
import random
import numpy as np
import torch
from utils import helpers as utl
import matplotlib.pyplot as plt
import seaborn as sns

from gym import Env
from gym import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def rgb2gray(rgb):
    gray = 0.2989 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return np.expand_dims(gray, axis=0)


def semi_circle_goal_sampler():
    r = 1.0
    angle = random.uniform(0, np.pi)
    goal = r * np.array((np.cos(angle), np.sin(angle)))
    return goal


def circle_goal_sampler():
    r = 1.0
    angle = random.uniform(0, 2 * np.pi)
    goal = r * np.array((np.cos(angle), np.sin(angle)))
    return goal


GOAL_SAMPLERS = {
    'semi-circle': semi_circle_goal_sampler,
    'circle': circle_goal_sampler,
}


class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self, max_episode_steps=100, goal_sampler=None):
        if callable(goal_sampler):
            self.goal_sampler = goal_sampler
        elif isinstance(goal_sampler, str):
            self.goal_sampler = GOAL_SAMPLERS[goal_sampler]
        elif goal_sampler is None:
            self.goal_sampler = semi_circle_goal_sampler
        else:
            raise NotImplementedError(goal_sampler)

        self.reset_task()
        self.task_dim = 2
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            # shape=((shp[0] * self.num_frames,) + shp[1:]),
            shape=(1, 64, 64),
            dtype=np.uint8
        )
        # we convert the actions from [-1, 1] to [-0.1, 0.1] in the step() function
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self._max_episode_steps = max_episode_steps

    def sample_task(self):
        goal = self.goal_sampler()
        return goal

    def set_task(self, task):
        self._goal = task

    def get_task(self):
        return self._goal

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_task()
        self.set_task(task)
        return task

    def reset_model(self):
        self._state = np.zeros(2)
        return self.obs_to_image(self._get_obs())

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def get_state(self):
        return self._state

    # def obs_to_image(self, obs):
    #     fig = plt.figure(figsize=(6.4, 6.4), dpi=10)
    #     canvas = FigureCanvasAgg(fig)
    #     # plot half circle and goal position
    #     angles = np.linspace(0, np.pi, num=100)
    #     x, y = np.cos(angles), np.sin(angles)
    #     plt.plot(x, y, '--', color='k')
    #     # plt.scatter(*tuple(self._goal), marker='x', color='r', s=80)
    #     # fix visualization
    #     plt.axis('scaled')
    #     # ax.set_xlim(-1.25, 1.25)
    #     plt.xlim(-2, 2)
    #     # ax.set_ylim(-0.25, 1.25)
    #     plt.ylim(-1, 2)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.axis('off')
    #     plt.scatter(obs[0], obs[1], marker='.', s=50, c='r')
    #     fig.tight_layout(pad=0)
    #     plt.margins(0)
    #     canvas.draw()
    #     buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    #     buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))#[110:-110,75:-75,:]
    #     plt.close()
    #     return buf.transpose([-1,0,1])#rgb2gray(buf)

    def obs_to_image(self, obs):
        mat = np.zeros((1, 64, 64))
        x = np.round((2 + obs[0]) * 16).astype(np.int)
        y = np.round(-1 * ((obs[1] - 2) / 3) * 64).astype(np.int)
        if x >= 0 and x <= 63 and y >= 0 and y <= 63:
            mat[:, y, x] = 1
        return mat

    def step(self, action):

        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action), action

        self._state = self._state + 0.1 * action
        reward = - np.linalg.norm(self._state - self._goal, ord=2)
        done = False
        ob = self._get_obs()
        info = {'task': self.get_task(), 'state': self._state}
        return self.obs_to_image(ob), reward, done, info

    def visualise_behaviour(self,
                            env,
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
        episode_prev_pstate = [[] for _ in range(num_episodes)]
        episode_next_pstate = [[] for _ in range(num_episodes)]
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
        start_pstate = env.venv.envs[0].get_state()
        start_obs_raw = state.clone()
        task = task.view(-1) if task is not None else None

        # initialise actions and rewards (used as initial input to policy if we have a recurrent policy)
        if hasattr(args, 'hidden_size'):
            hidden_state = torch.zeros((1, args.hidden_size)).to(device)
        else:
            hidden_state = None

        # keep track of what task we're in and the position of the cheetah
        pos = [[] for _ in range(args.max_rollouts_per_task)]
        start_pos = state

        for episode_idx in range(num_episodes):

            curr_rollout_rew = []
            pos[episode_idx].append(start_pos[0])

            if episode_idx == 0:
                if encoder is not None:
                    # reset to prior
                    current_hidden_state = encoder.prior(1)
                    current_hidden_state.to(device)
            episode_hidden_states[episode_idx].append(current_hidden_state[0].clone())

            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    episode_prev_obs[episode_idx].append(start_obs_raw.clone())
                    episode_prev_pstate[episode_idx].append(start_pstate.copy())
                else:
                    episode_prev_obs[episode_idx].append(state.clone())
                    episode_prev_pstate[episode_idx].append(pstate.copy())
                    # act
                _, action = utl.select_action_cpc(args=args, policy=policy, deterministic=True,
                                                  hidden_latent=current_hidden_state.squeeze(0), state=state, task=task)

                (state, task), (rew, rew_normalised), done, info = utl.env_step(env, action, args)
                # state = state.float().reshape((1, -1)).to(device)
                pstate = info[0]['state'].copy()
                task = task.view(-1) if task is not None else None

                # keep track of position
                pos[episode_idx].append(state[0])

                if encoder is not None:
                    # update task embedding
                    current_hidden_state = encoder(
                        action.reshape(1, -1).float().to(device), state, rew.reshape(1, -1).float().to(device),
                        current_hidden_state, return_prior=False)

                    episode_hidden_states[episode_idx].append(current_hidden_state[0].clone())
                episode_next_pstate[episode_idx].append(pstate.copy())
                episode_next_obs[episode_idx].append(state.clone())
                episode_rewards[episode_idx].append(rew.clone())
                episode_actions[episode_idx].append(action.clone())

                if info[0]['done_mdp'] and not done:
                    start_obs_raw = info[0]['start_state']
                    start_obs_raw = torch.from_numpy(start_obs_raw).float().to(device).unsqueeze(0)
                    start_pos = start_obs_raw
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)

        # clean up
        if encoder is not None:
            episode_hidden_states = [torch.stack(e) for e in episode_hidden_states]
        episode_prev_state = [np.vstack(episode) for episode in episode_prev_pstate]
        episode_next_state = [np.vstack(episode) for episode in episode_next_pstate]
        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.stack(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]

        figsize = (5.5, 4)
        figure, axis = plt.subplots(1, 1, figsize=figsize)
        xlim = (-1.3, 1.3)
        if self.goal_sampler == semi_circle_goal_sampler:
            ylim = (-0.3, 1.3)
        else:
            ylim = (-1.3, 1.3)
        color_map = mpl.colors.ListedColormap(sns.color_palette("husl", num_episodes))

        observations = np.stack([np.array(episode_prev_pstate[i]) for i in range(num_episodes)])
        curr_task = env.get_task()[0]

        # plot goal
        axis.scatter(curr_task[0], curr_task[1], marker='x', color='k', s=50)
        # radius where we get reward
        if hasattr(self, 'goal_radius'):
            circle1 = plt.Circle(curr_task, self.goal_radius, color='c', alpha=0.2, edgecolor='none')
            plt.gca().add_artist(circle1)

        for i in range(num_episodes):
            color = color_map(i)
            path = observations[i]

            # plot (semi-)circle
            r = 1.0
            if self.goal_sampler == semi_circle_goal_sampler:
                angle = np.linspace(0, np.pi, 100)
            else:
                angle = np.linspace(0, 2 * np.pi, 100)
            goal_range = r * np.array((np.cos(angle), np.sin(angle)))
            plt.plot(goal_range[0], goal_range[1], 'k--', alpha=0.1)

            # plot trajectory
            axis.plot(path[:, 0], path[:, 1], '-', color=color, label=i)
            axis.scatter(*path[0, :2], marker='.', color=color, s=50)
            if i == 1 and kwargs['belief_evaluator'] is not None:
                r = 1
                num_points_semicircle = 150
                angles = np.linspace(0, np.pi, num=num_points_semicircle)
                x, y = r * np.cos(angles), r * np.sin(angles)
                belief = 1. - torch.sigmoid(
                    kwargs['belief_evaluator'](episode_hidden_states[i][0])).detach().cpu().numpy().flatten()
                plt.scatter(x, y, c=belief, cmap='gray')

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks([])
        plt.yticks([])
        plt.legend()
        plt.tight_layout()
        kwargs['logger'].add_figure('belief', figure, iter_idx)
        if image_folder is not None:
            plt.savefig('{}/{}_behaviour.png'.format(image_folder, iter_idx), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        # j = -2#np.random.randint(20,80)
        # angles = np.linspace(0, np.pi, num=100)
        # x, y = np.cos(angles), np.sin(angles)
        # circ_points = torch.from_numpy(np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])).to(device)
        # fig, ax = plt.subplots(1, num_episodes, figsize=(figsize[0]*3, figsize[1]))
        # for i in range(num_episodes):
        #     belief = torch.nn.Softmax()(kwargs['reward_decoder'](episode_hidden_states[i][j]))
        #     ax[i].plot(observations[i,:j+1, 0], observations[i,:j+1, 1], '-', color=color, label=0)
        #     ax[i].scatter(*observations[i, 0, :2], marker='.', color=color, s=50)
        #     plt.gray()
        #     ax[i].scatter(x, y, c=belief.detach().cpu().numpy().flatten())
        #     ax[i].scatter(*curr_task, marker='x', color='r', s=80)
        #     if hasattr(self, 'goal_radius'):
        #         circle1 = plt.Circle(curr_task, self.goal_radius, color='c', alpha=0.2, edgecolor='none')
        #         ax[i].add_artist(circle1)
        #     ax[i].set_xlim(xlim)
        #     ax[i].set_ylim(ylim)
        #     ax[i].set_xticks([])
        #     ax[i].set_yticks([])
        #
        # #plt.tight_layout()
        # kwargs['logger'].add_figure('belief', fig, iter_idx)

        plt_rew = [episode_rewards[i][:episode_lengths[i]] for i in range(len(episode_rewards))]
        plt.plot(torch.cat(plt_rew).view(-1).cpu().numpy())
        plt.xlabel('env step')
        plt.ylabel('reward per step')

        plt.tight_layout()

        if image_folder is not None:
            plt.savefig('{}/{}_rewards.png'.format(image_folder, iter_idx), dpi=300, bbox_inches='tight')
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

    def visualise_behaviour_for_script(self,
                            env,
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
        episode_prev_pstate = [[] for _ in range(num_episodes)]
        episode_next_pstate = [[] for _ in range(num_episodes)]
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
        start_pstate = env.venv.envs[0].get_state()
        start_obs_raw = state.clone()
        task = task.view(-1) if task is not None else None

        # initialise actions and rewards (used as initial input to policy if we have a recurrent policy)
        if hasattr(args, 'hidden_size'):
            hidden_state = torch.zeros((1, args.hidden_size)).to(device)
        else:
            hidden_state = None

        # keep track of what task we're in and the position of the cheetah
        pos = [[] for _ in range(args.max_rollouts_per_task)]
        start_pos = state

        for episode_idx in range(num_episodes):

            curr_rollout_rew = []
            pos[episode_idx].append(start_pos[0])

            if episode_idx == 0:
                if encoder is not None:
                    # reset to prior
                    current_hidden_state = encoder.prior(1)
                    current_hidden_state.to(device)
            episode_hidden_states[episode_idx].append(current_hidden_state[0].clone())

            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    episode_prev_obs[episode_idx].append(start_obs_raw.clone())
                    episode_prev_pstate[episode_idx].append(start_pstate.copy())
                else:
                    episode_prev_obs[episode_idx].append(state.clone())
                    episode_prev_pstate[episode_idx].append(pstate.copy())
                    # act
                _, action = utl.select_action_cpc(args=args, policy=policy, deterministic=True,
                                                  hidden_latent=current_hidden_state.squeeze(0), state=state, task=task)

                (state, task), (rew, rew_normalised), done, info = utl.env_step(env, action, args)
                # state = state.float().reshape((1, -1)).to(device)
                pstate = info[0]['state'].copy()
                task = task.view(-1) if task is not None else None

                # keep track of position
                pos[episode_idx].append(state[0])

                if encoder is not None:
                    # update task embedding
                    current_hidden_state = encoder(
                        action.reshape(1, -1).float().to(device), state, rew.reshape(1, -1).float().to(device),
                        current_hidden_state, return_prior=False)

                    episode_hidden_states[episode_idx].append(current_hidden_state[0].clone())
                episode_next_pstate[episode_idx].append(pstate.copy())
                episode_next_obs[episode_idx].append(state.clone())
                episode_rewards[episode_idx].append(rew.clone())
                episode_actions[episode_idx].append(action.clone())

                if info[0]['done_mdp'] and not done:
                    start_obs_raw = info[0]['start_state']
                    start_obs_raw = torch.from_numpy(start_obs_raw).float().to(device).unsqueeze(0)
                    start_pos = start_obs_raw
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)

        # clean up
        if encoder is not None:
            episode_hidden_states = [torch.stack(e) for e in episode_hidden_states]
        episode_prev_state = [np.vstack(episode) for episode in episode_prev_pstate]
        episode_next_state = [np.vstack(episode) for episode in episode_next_pstate]
        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.stack(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]

        figsize = (5.5, 4)
        figure, axis = plt.subplots(1, 1, figsize=figsize)
        xlim = (-1.3, 1.3)
        if self.goal_sampler == semi_circle_goal_sampler:
            ylim = (-0.3, 1.3)
        else:
            ylim = (-1.3, 1.3)
        color_map = mpl.colors.ListedColormap(sns.color_palette("husl", num_episodes))

        observations = np.stack([np.array(episode_prev_pstate[i]) for i in range(num_episodes)])
        curr_task = env.get_task()[0]

        # plot goal
        axis.scatter(curr_task[0], curr_task[1], marker='x', color='k', s=50)
        # radius where we get reward
        if hasattr(self, 'goal_radius'):
            circle1 = plt.Circle(curr_task, self.goal_radius, color='c', alpha=0.2, edgecolor='none')
            plt.gca().add_artist(circle1)

        for i in range(num_episodes):
            color = color_map(i)
            path = observations[i]

            # plot (semi-)circle
            r = 1.0
            if self.goal_sampler == semi_circle_goal_sampler:
                angle = np.linspace(0, np.pi, 100)
            else:
                angle = np.linspace(0, 2 * np.pi, 100)
            goal_range = r * np.array((np.cos(angle), np.sin(angle)))
            plt.plot(goal_range[0], goal_range[1], 'k--', alpha=0.1)

            # plot trajectory
            axis.plot(path[:, 0], path[:, 1], '-', color=color, label=i)
            axis.scatter(*path[0, :2], marker='.', color=color, s=50)
            # if i == 1 and kwargs['belief_evaluator'] is not None:
            #     r = 1
            #     num_points_semicircle = 150
            #     angles = np.linspace(0, np.pi, num=num_points_semicircle)
            #     x, y = r * np.cos(angles), r * np.sin(angles)
            #     belief = 1. - torch.sigmoid(
            #         kwargs['belief_evaluator'](episode_hidden_states[i][0])).detach().cpu().numpy().flatten()
            #     plt.scatter(x, y, c=belief, cmap='gray')
        #
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks([])
        plt.yticks([])
        plt.legend()
        plt.tight_layout()
        # kwargs['logger'].add_figure('belief', figure, iter_idx)
        if image_folder is not None:
            plt.savefig('{}/{}_behaviour.png'.format(image_folder, iter_idx), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        # j = -2#np.random.randint(20,80)
        # angles = np.linspace(0, np.pi, num=100)
        # x, y = np.cos(angles), np.sin(angles)
        # circ_points = torch.from_numpy(np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])).to(device)
        # fig, ax = plt.subplots(1, num_episodes, figsize=(figsize[0]*3, figsize[1]))
        # for i in range(num_episodes):
        #     belief = torch.nn.Softmax()(kwargs['reward_decoder'](episode_hidden_states[i][j]))
        #     ax[i].plot(observations[i,:j+1, 0], observations[i,:j+1, 1], '-', color=color, label=0)
        #     ax[i].scatter(*observations[i, 0, :2], marker='.', color=color, s=50)
        #     plt.gray()
        #     ax[i].scatter(x, y, c=belief.detach().cpu().numpy().flatten())
        #     ax[i].scatter(*curr_task, marker='x', color='r', s=80)
        #     if hasattr(self, 'goal_radius'):
        #         circle1 = plt.Circle(curr_task, self.goal_radius, color='c', alpha=0.2, edgecolor='none')
        #         ax[i].add_artist(circle1)
        #     ax[i].set_xlim(xlim)
        #     ax[i].set_ylim(ylim)
        #     ax[i].set_xticks([])
        #     ax[i].set_yticks([])
        #
        # #plt.tight_layout()
        # kwargs['logger'].add_figure('belief', fig, iter_idx)

        plt_rew = [episode_rewards[i][:episode_lengths[i]] for i in range(len(episode_rewards))]
        plt.plot(torch.cat(plt_rew).view(-1).cpu().numpy())
        plt.xlabel('env step')
        plt.ylabel('reward per step')

        plt.tight_layout()

        if image_folder is not None:
            plt.savefig('{}/{}_rewards.png'.format(image_folder, iter_idx), dpi=300, bbox_inches='tight')
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


class SparsePointEnv(PointEnv):
    """ Reward is L2 distance given only within goal radius """

    def __init__(self, goal_radius=0.2, max_episode_steps=100, goal_sampler='semi-circle'):
        super().__init__(max_episode_steps=max_episode_steps, goal_sampler=goal_sampler)
        self.goal_radius = goal_radius
        self.reset_task()

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r

    def reset_model(self):
        self._state = np.array([0, 0])
        return self.obs_to_image(self._get_obs())

    def calc_reward_for_state(self):
        return self.sparsify_rewards(- np.linalg.norm(self._state - self._goal, ord=2))

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        d.update({'sparse_reward': sparse_reward})
        d.update({'dense_reward': reward})
        return ob, sparse_reward, done, d
