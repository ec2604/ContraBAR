import matplotlib.pyplot as plt
import numpy as np
import torch
from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_

from utils import helpers as utlutl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HalfCheetahEnv(HalfCheetahEnv_):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    def viewer_setup(self):
        camera_id = self.model.camera_name2id('track')
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        # Hide the overlay
        self.viewer._hide_overlay = True

    def render(self, mode='human', width=500, height=500):
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            return data[::-1]
        elif mode == 'human':
            self._get_viewer().render()

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
        state, belief, task = utl.reset_env(env, args)
        start_state = state.clone()

        if hasattr(args, 'hidden_size'):
            hidden_state = torch.zeros((1, args.hidden_size)).to(device)
        else:
            hidden_state = None

        # keep track of what task we're in and the position of the cheetah
        pos = [[] for _ in range(args.max_rollouts_per_task)]
        start_pos = unwrapped_env.get_body_com("torso")[0].copy()

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
                    episode_prev_obs[episode_idx].append(start_state.clone())
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
                state = state.reshape((1, -1)).float().to(device)

                # keep track of position
                pos[episode_idx].append(unwrapped_env.get_body_com("torso")[0].copy())

                if encoder is not None:
                    # update task embedding
                    current_hidden_state = encoder(
                        action.reshape(1, -1).float().to(device), state, rew.reshape(1, -1).float().to(device),
                        hidden_state, return_prior=False)

                    episode_hidden_states[episode_idx].append(current_hidden_state[0].clone())

                episode_next_obs[episode_idx].append(state.clone())
                episode_rewards[episode_idx].append(rew.clone())
                episode_actions[episode_idx].append(action.reshape(1, -1).clone())

                if info[0]['done_mdp'] and not done:
                    start_state = info[0]['start_state']
                    start_state = torch.from_numpy(start_state).reshape((1, -1)).float().to(device)
                    start_pos = unwrapped_env.get_body_com("torso")[0].copy()
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)


        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.cat(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]

        # plot the movement of the half-cheetah
        plt.figure(figsize=(7, 4 * num_episodes))
        min_x = min([min(p) for p in pos])
        max_x = max([max(p) for p in pos])
        span = max_x - min_x
        for i in range(num_episodes):
            plt.subplot(num_episodes, 1, i + 1)
            # (not plotting the last step because this gives weird artefacts)
            plt.plot(pos[i][:-1], range(len(pos[i][:-1])), 'k')
            plt.title('task: {}'.format(task), fontsize=15)
            plt.ylabel('steps (ep {})'.format(i), fontsize=15)
            if i == num_episodes - 1:
                plt.xlabel('position', fontsize=15)
            # else:
            #     plt.xticks([])
            plt.xlim(min_x - 0.05 * span, max_x + 0.05 * span)
            plt.plot([0, 0], [200, 200], 'b--', alpha=0.2)
        plt.tight_layout()
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
