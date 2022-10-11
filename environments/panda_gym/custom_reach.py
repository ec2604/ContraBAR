import numpy as np

from panda_gym.envs.core import Task, RobotTaskEnv
from panda_gym.utils import distance
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.reach import Reach
from panda_gym.pybullet import PyBullet
from utils import helpers as utl
from mpl_toolkits import mplot3d
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import random
import gym
import cv2
import gym.spaces
import gym.utils.seeding
import numpy as np
import torch
import pybullet
import pybullet_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomReach(Task):
    def __init__(
            self,
            sim,
            get_ee_position,
            reward_type="sparse",
            distance_threshold=0.05,
            goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal = self._sample_goal()
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, goal_range / 2])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range / 2])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.05,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=None)  # np.array([0.1, 0.9, 0.1, 0.3]),

    #  )

    def get_obs(self) -> np.ndarray:
        return np.array([])

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def reset_task(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        # goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        r = 0.15
        theta = random.random() * (np.pi) - (np.pi / 2)
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        z = 0.15 / 2
        goal = np.array([x, y, z])
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray):
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = distance(achieved_goal, desired_goal)
        r = None
        if self.reward_type == "sparse":
            r = float(d < self.distance_threshold)# + float(d > self.distance_threshold)*(-1)
        else:
            r = -d
        return r


class CustomReachEnv(RobotTaskEnv):
    """Robotic task goal env, as the junction of a task and a robot.
    Args:
        robot (PyBulletRobot): The robot.
        task (task): The task.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "joints") -> None:
        sim = PyBullet(render=render)

        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = CustomReach(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position)
        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."
        self.sim = robot.sim
        self.robot = robot
        self.task = task
        self.task_dim = 3
        self._max_episode_steps = 50
        obs = self.reset()  # required for init; seed can be changed later
        observation_shape = obs.shape
        # achieved_goal_shape = obs["achieved_goal"].shape
        # desired_goal_shape = obs["achieved_goal"].shape
        self.observation_space = gym.spaces.Box(
               0, 255, shape=observation_shape, dtype=np.float32)
                # desired_goal=gym.spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
                # achieved_goal=gym.spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32)
        self.action_space = self.robot.action_space
        self.compute_reward = self.task.compute_reward
        self._saved_goal = dict()

    def _get_obs(self):
        observation = self.render('rgb_array', 84, 84)
        if self.task.is_success(self.task.get_achieved_goal(), self.task.get_goal()):
            observation[-20:,:,:] = 1

        # achieved_goal = self.task.get_achieved_goal()
        # return {
        #     "observation": observation,
        #     "achieved_goal": achieved_goal,
        #     "desired_goal": self.task.get_goal(),
        # }
        observation = np.transpose(observation, axes=[2, 0, 1])
        return observation

    def reset(self, seed=None):
        self.task.np_random, seed = gym.utils.seeding.np_random(seed)
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        return self._get_obs()

    def get_task(self):
        """
        Return a task description, such as goal position or target velocity.
        """
        return self.task.goal

    def reset_task(self, seed=None):
        self.task.np_random, seed = gym.utils.seeding.np_random(seed)
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset_task()
        return self._get_obs()

    def save_state(self) -> int:
        state_id = self.sim.save_state()
        self._saved_goal[state_id] = self.task.goal
        return state_id

    def restore_state(self, state_id: int) -> None:
        self.sim.restore_state(state_id)
        self.task.goal = self._saved_goal[state_id]

    def remove_state(self, state_id: int) -> None:
        self._saved_goal.pop(state_id)
        self.sim.remove_state(state_id)

    def step(self, action: np.ndarray):
        act = action
        self.robot.set_action(act)
        self.sim.step()
        obs = self._get_obs()
        done = False
        info = {"is_success": self.task.is_success(self.task.get_achieved_goal(), self.task.get_goal())}
        reward = self.task.compute_reward(self.task.get_achieved_goal(), self.task.get_goal(), info)# - 0.1*np.linalg.norm(action)
        assert isinstance(reward, float)  # needed for pytype cheking
        return obs, reward, done, info

    def close(self) -> None:
        self.sim.close()

    def render(
            self,
            mode: str = 'rgb_array',
            width: int = 720,
            height: int = 480,
            target_position=None,
            distance: float = 1.4,
            yaw: float = 45,
            pitch: float = -30,
            roll: float = 0,
    ):
        """Render.
        If mode is "human", make the rendering real-time. All other arguments are
        unused. If mode is "rgb_array", return an RGB array of the scene.
        Args:
            mode (str): "human" of "rgb_array". If "human", this method waits for the time necessary to have
                a realistic temporal rendering and all other args are ignored. Else, return an RGB array.
            width (int, optional): Image width. Defaults to 720.
            height (int, optional): Image height. Defaults to 480.
            target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
                Defaults to [0., 0., 0.].
            distance (float, optional): Distance of the camera. Defaults to 1.4.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -30.
            roll (int, optional): Rool of the camera. Defaults to 0.
        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        target_position = target_position if target_position is not None else np.zeros(3)
        return self.sim.render(
            mode,
            width=width,
            height=height,
            target_position=target_position,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )

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
        #task = task.view(-1) if task is not None else None

        # initialise actions and rewards (used as initial input to policy if we have a recurrent policy)
        if hasattr(args, 'hidden_size'):
            hidden_state = torch.zeros((1, args.hidden_size)).to(device)
        else:
            hidden_state = None

        # keep track of what task we're in and the position of the cheetah
        pos = [[] for _ in range(args.max_rollouts_per_task)]
        imgs = [[] for _ in range(args.max_rollouts_per_task)]
        start_pos = unwrapped_env.task.get_achieved_goal()
        unwrapped_env.sim.physics_client.changeVisualShape(unwrapped_env.sim._bodies_idx['target'], -1,
                                                           rgbaColor=np.array([0.1, 0.9, 0.1, 0.3]))
        start_img = unwrapped_env.render()
        unwrapped_env.sim.physics_client.changeVisualShape(unwrapped_env.sim._bodies_idx['target'], -1,
                                                           rgbaColor=np.array([0, 0, 0, 0]))
        for episode_idx in range(num_episodes):

            curr_rollout_rew = []
            pos[episode_idx].append(start_pos)
            imgs[episode_idx].append(start_img)
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
                # underlying_state = torch.from_numpy(unwrapped_env.render_()).unsqueeze(0).to(device)
                # if done[0]:
                #     underlying_state = torch.cat([underlying_state, torch.ones(underlying_state.shape[0], 1,
                #                                                                *underlying_state.shape[2:]).to(device)],
                #                                  dim=1).float()
                # else:
                #     underlying_state = torch.cat([underlying_state, torch.zeros(underlying_state.shape[0], 1,
                #                                                                 *underlying_state.shape[2:]).to(
                #         device)],
                #                                  dim=1).float()
                # task = task.view(-1) if task is not None else None

                # keep track of position
                pos[episode_idx].append(unwrapped_env.task.get_achieved_goal())
                unwrapped_env.sim.physics_client.changeVisualShape(unwrapped_env.sim._bodies_idx['target'], -1,
                                                                   rgbaColor=np.array([0.1, 0.9, 0.1, 0.3]))
                curr_img = unwrapped_env.render()
                if rew > 0:
                    curr_img[-20:,:,:] = 1
                imgs[episode_idx].append(curr_img)
                unwrapped_env.sim.physics_client.changeVisualShape(unwrapped_env.sim._bodies_idx['target'], -1,
                                                                   rgbaColor=np.array([0, 0, 0, 0]))
                if encoder is not None:
                    # update task embedding
                    current_hidden_state = encoder(
                        action.reshape(1, -1).float().to(device), state,
                        rew.reshape(1, -1).float().to(device),
                        current_hidden_state, return_prior=False)

                    episode_hidden_states[episode_idx].append(current_hidden_state[0].clone())

                episode_next_obs[episode_idx].append(state.clone())
                episode_rewards[episode_idx].append(rew.clone())
                episode_actions[episode_idx].append(action.clone())

                if info[0]['done_mdp'] and not done:
                    start_obs_raw = info[0]['start_state']
                    start_obs_raw = torch.from_numpy(start_obs_raw).float().to(device).unsqueeze(0)
                    start_pos = unwrapped_env.task.get_achieved_goal().copy()
                    start_img = unwrapped_env.render()
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)

        # clean up

        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_hidden_states = torch.cat([torch.cat(e)[:49] for e in episode_hidden_states]).unsqueeze(1)
        episode_actions = [torch.stack(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]
        pos_episodes = np.concatenate([np.array(p)[:49, :2] for p in pos])
        # kwargs['logger'].add_video('behaviour_video_rollout_1', np.transpose(np.stack([imgs[0][:-1]]), axes=[0, 1, 4, 2, 3]),
        #                            iter_idx)
        # kwargs['logger'].add_video('behaviour_video_rollout_2', np.transpose(np.stack([imgs[1][:-1]]), axes=[0, 1, 4, 2, 3]),
        #                            iter_idx)
        # kwargs['logger'].add_video('behaviour_video_rollout_3', np.transpose(np.stack([imgs[2][:-1]]), axes=[0, 1, 4, 2, 3]),
        #                            iter_idx)
        #
        # curr_task = torch.tensor(env.get_task(),device=device)
        # predictor_input, _, (x,y,z) = utl.generate_predictor_input(episode_hidden_states, curr_task)
        # rewards = torch.sigmoid(kwargs['belief_evaluator'](predictor_input)).squeeze(1)
        # thetas = np.linspace(-np.pi / 2, np.pi / 2, 1000)
        # r = 0.15
        # sc_x = r*np.cos(thetas)
        # sc_y = r*np.sin(thetas)
        # dx = (x[1] - x[0]) / 2.
        # dy = (y[1] - y[0]) / 2.
        # extent = [x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy]
        # belief_video = []
        # for i in range(len(pos_episodes)):
        # for i in range(rewards.shape[0]):
        #     fig, ax = plt.subplots()
            # plt.imshow(rewards[i].reshape(len(x),len(x)).detach().cpu().numpy(), extent=extent, cmap='gray', vmin=0, vmax=1, origin='lower')
            # plt.colorbar()
            # ax.add_patch(plt.Circle(xy=curr_task[0][:2].detach().cpu().numpy(), radius=0.05, alpha=0.4, color='g'))
            # x_pos, y_pos = pos_episodes[(i//49)*49:i+1, 0 ], pos_episodes[49*(i // 49) :i+1, 1]
            # plt.plot(sc_x, sc_y, alpha=0.5, c='m',)
            # plt.plot(x_pos, y_pos, alpha=0.8,label=f'Episode {i // 50}, step: {i}')
            # plt.legend()
            # canvas = FigureCanvas(fig)
            # canvas.draw()
            # buf = canvas.buffer_rgba()
            # belief_arr = np.asarray(buf)
            # belief_video.append(belief_arr)
            # plt.close()
        # fps = 10
        # belief_video = np.stack(belief_video)
        # size = belief_video.shape[1:3]
        # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        # print("saving to: ", image_folder + '/test_seed_43.avi', flush=True)
        # out = cv2.VideoWriter(image_folder + '/test_seed_43.avi', fourcc, fps, (size[1], size[0]))
        # for img in belief_video:
        #     img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        #     out.write(img)
        # out.release()
        # video = np.expand_dims(np.stack(belief_video), 0)
        # video =   np.transpose(video, axes=[0, 1, 4, 2, 3])
        # kwargs['logger'].add_video('belief', video, iter_idx)
        # kwargs['logger'].add_video('behaviour_video_rollout_1',
        #                            episode_prev_obs[0][:, 2, ...].unsqueeze(0).unsqueeze(2).to(torch.uint8), iter_idx)
        # kwargs['logger'].add_video('behaviour_video_rollout_2',
        #                            episode_prev_obs[1][:, 2, ...].unsqueeze(0).unsqueeze(2).to(torch.uint8), iter_idx)
        # plot the movement of the ant
        # print(pos)
        #
        # for i in range(num_episodes):
        #     figure = plt.figure()
        #     ax = plt.axes(projection='3d')
        #
        #     x = list(map(lambda p: p[0], pos[i]))
        #     y = list(map(lambda p: p[1], pos[i]))
        #     z = list(map(lambda p: p[2], pos[i]))
        #     ax.scatter3D(x[0], y[0], z[0], 'bo')
        #
        #     ax.plot3D(x[:-1], y[:-1], z[:-1], 'g')
        #
        #     curr_task = env.get_task()[0]
        #     plt.title('task: {}'.format(curr_task), fontsize=15)
        #     ax.scatter3D(curr_task[0], curr_task[1], curr_task[2], 'rx')
        #     # u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        #     # r = unwrapped_env.task.distance_threshold
        #     # x = r*np.cos(u) * np.sin(v)
        #     # y = r*np.sin(u) * np.sin(v)
        #     # z = r*np.cos(v)
        #     # ax.plot_surface(curr_task[0] + x, curr_task[1] + y, curr_task[2] + z, alpha=0.5)
        #     plt.tight_layout()
        #     kwargs['logger'].add_figure(f'belief_{i}', figure, iter_idx)
        #     # if image_folder is not None:
        #     #     plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
        #     #     plt.close()
        #     # else:
        #     #     plt.show()

        if not return_pos:
            return episode_hidden_states, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns
        else:
            return episode_hidden_states, \
                   episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns, pos