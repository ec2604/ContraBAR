import matplotlib.pyplot as plt
import numpy as np
import torch
from random import random
from environments.mujoco.mujoco_env import MujocoEnv
from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def rgb2gray(rgb):
    gray = 0.2989*rgb[...,0]+0.587*rgb[...,1]+0.114*rgb[...,2]
    return np.expand_dims(gray, axis=0)

class AntEnv(MujocoEnv):
    def __init__(self, use_low_gear_ratio=False):
        self.init_serialization(locals())
        if use_low_gear_ratio:
            xml_path = 'low_gear_ratio_ant.xml'
        else:
            xml_path = 'ant.xml'
        super().__init__(
            xml_path,
            frame_skip=5,
            automatically_set_obs_and_action_space=True,
        )

    def step(self, a):
        torso_xyz_before = self.get_body_com("torso")
        self.do_simulation(a, self.frame_skip)
        torso_xyz_after = self.get_body_com("torso")
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = torso_velocity[0] / self.dt
        ctrl_cost = 0.  # .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.  # 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )

    def _get_obs(self):
        # this is gym ant obs, should use rllab?
        # if position is needed, override this in subclasses
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def reset_task(self, task):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

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
        # episode_prev_img = [[] for _ in range(num_episodes)]
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
        # start_img = rgb2gray(self.render('rgb_array', height=64, width=64))
        #task = task.view(-1) if task is not None else None

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
                    # episode_prev_img[episode_idx].append(start_img.copy())
                else:
                    episode_prev_obs[episode_idx].append(state.clone())
                    # episode_prev_img[episode_idx].append(curr_img.copy())
                # act
                _, action = utl.select_action_cpc(args=args, policy=policy, deterministic=True,
                                                  hidden_latent=current_hidden_state.squeeze(0), state=state, task=task)


                (state, task), (rew, rew_normalised), done, info = utl.env_step(env, action, args)
                # curr_img = rgb2gray(self.render('rgb_array', height=64, width=64))
                state = state.float().reshape((1, -1)).to(device)
                # rew_cpc = torch.from_numpy(np.array(info[0]['goal_forward'] >= 0.3))
                # task = task.view(-1) if task is not None else None

                # keep track of position
                pos[episode_idx].append(unwrapped_env.get_body_com("torso")[:2].copy())

                if encoder is not None:
                    # update task embedding
                    current_hidden_state = encoder(
                        action.reshape(1, -1).float().to(device), state,
                        # rew_cpc.reshape(1, -1).float().to(device), #
                        rew.reshape(1, -1).float().to(device),
                        current_hidden_state, return_prior=False)

                    episode_hidden_states[episode_idx].append(current_hidden_state[0].clone())

                episode_next_obs[episode_idx].append(state.clone())
                episode_rewards[episode_idx].append(rew.clone())
                episode_actions[episode_idx].append(action.clone())

                if info[0]['done_mdp'] and not done:
                    start_obs_raw = info[0]['start_state']
                    start_obs_raw = torch.from_numpy(start_obs_raw).float().reshape((1, -1)).to(device)
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
        #                            episode_prev_img[0].unsqueeze(0).unsqueeze(2).to(torch.uint8), iter_idx)
        # plot the movement of the ant
        # print(pos)

        for i in range(num_episodes):
            figure = plt.figure()
            x = list(map(lambda p: p[0], pos[i]))[:-1]
            y = list(map(lambda p: p[1], pos[i]))[:-1]
            angle = np.linspace(0, np.pi)
            goal_range_x = np.cos(angle)
            goal_range_y = np.sin(angle)
            plt.plot(goal_range_x, goal_range_y, 'k--', alpha=0.1)
            plt.plot(x[0], y[0], 'bo')

            plt.plot(x, y, '-')
            curr_task = env.get_task()[0]
            plt.title(f'task: {curr_task}, Cumulative reward: {episode_rewards[i].sum():.2f}')
            if 'Goal' in args.env_name:
                plt.plot(curr_task[0], curr_task[1], 'rx')
            if hasattr(self, 'goal_radius'):
                circle1 = plt.Circle(curr_task, self.goal_radius, color='c', alpha=0.2, edgecolor='none')
                plt.gca().add_artist(circle1)

            plt.ylabel('y-position (ep {})'.format(i), fontsize=15)

            plt.xlabel('x-position', fontsize=15)
            plt.ylabel('y-position (ep {})'.format(i), fontsize=15)
            min_dim = -3.5
            max_dim = 3.5
            span = max_dim - min_dim
            plt.xlim(min_dim - 0.05 * span, max_dim + 0.05 * span)
            plt.ylim(min_dim - 0.05 * span, max_dim + 0.05 * span)

            plt.tight_layout()
            kwargs['logger'].add_figure(f'belief_{i}', figure, iter_idx)
            plt.close()
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
