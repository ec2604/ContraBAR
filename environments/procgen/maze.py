import matplotlib as mpl
import random
import numpy as np
import torch
import random
from utils import helpers as utl
import matplotlib.pyplot as plt
import seaborn as sns
import gym


from gym import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def rgb2gray(rgb):
    gray = 0.2989*rgb[:,:,0]+0.587*rgb[:,:,1]+0.114*rgb[:,:,2]
    return np.expand_dims(gray, axis=0)

class MazeEnv(gym.Env):
    def __init__(self, max_episode_steps=100, **kwargs):
        super(MazeEnv, self).__init__()
        self.rand_seed = random.randint(1, 10000)
        self.env = gym.make('procgen:procgen-maze-v0', **kwargs, start_level=self.rand_seed)
        self.env_kwargs = kwargs
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(1, 64, 64),
            dtype=np.uint8
        )
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1
        self.action_space = spaces.Discrete(15)
    def step(self, action):
        if isinstance(action, int):
            next_obs, reward, done, infos = self.env.step(action)
        else:
            next_obs, reward, done, infos = self.env.step(action[0])
        next_obs = rgb2gray(next_obs)
        return next_obs, reward, done, infos

    def reset_task(self, task=None):
        if task is None:
            task = random.randint(1, 10000)
            self.rand_seed = task
            self.env = gym.make('procgen:procgen-maze-v0', **self.env_kwargs, start_level=self.rand_seed)
        self.env = gym.make('procgen:procgen-maze-v0', **self.env_kwargs, start_level=task)
        return task

    def reset(self):
        # action -1 forces reset
        return rgb2gray(self.env.step(-1)[0])

    def get_task(self):
        return np.array(self.rand_seed).reshape(-1, 1)

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
        state, belief, task = utl.reset_env(env, args)
        start_obs_raw = state.clone()
        task = task.view(-1) if task is not None else None

        # initialise actions and rewards (used as initial input to policy if we have a recurrent policy)
        if hasattr(args, 'hidden_size'):
            hidden_state = torch.zeros((1, args.hidden_size)).to(device)
        else:
            hidden_state = None

        # keep track of what task we're in and the position of the cheetah


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
                    episode_prev_obs[episode_idx].append(start_obs_raw.clone())
                else:
                    episode_prev_obs[episode_idx].append(state.clone())
                    # act
                _, action = utl.select_action_cpc(args=args,
                                                  policy=policy,
                                                  belief=belief,
                                                  task=task,
                                                  deterministic=True,
                                                  state=state,
                                                  hidden_latent=current_hidden_state.squeeze(0)
                                                  )

                (state, belief, task), (rew, rew_normalised), done, info = utl.env_step(env, action, args)
                #state = state.float().reshape((1, -1)).to(device)
                task = task.view(-1) if task is not None else None



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
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)

        # clean up
        if encoder is not None:
            episode_hidden_states = [torch.stack(e) for e in episode_hidden_states]
        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.stack(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]

        kwargs['logger'].add_video('behaviour_video_rollout_1',
                                   episode_prev_obs[0][:, 0, ...].unsqueeze(0).unsqueeze(2).to(torch.uint8), iter_idx)
        kwargs['logger'].add_video('behaviour_video_rollout_2',
                                   episode_prev_obs[1][:, 0, ...].unsqueeze(0).unsqueeze(2).to(torch.uint8), iter_idx)
        kwargs['logger'].add_video('behaviour_video_rollout_3',
                                   episode_prev_obs[2][:, 0, ...].unsqueeze(0).unsqueeze(2).to(torch.uint8), iter_idx)

        if not return_pos:
            return episode_hidden_states, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns
        else:
            return episode_hidden_states, \
                   episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns