import random

import numpy as np

from environments.mujoco.ant import AntEnv
import torch

class AntGoalEnv(AntEnv):
    def __init__(self, max_episode_steps=200):
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 2
        #self.wind = np.array([random.random() * 0.01 - 0.005,random.random() * 0.01 - 0.005])
        super(AntGoalEnv, self).__init__()


    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        #qpos = self.sim.data.qpos
        #qvel = self.sim.data.qvel
        #qpos[:2] += self.wind
        #self.set_state(qpos, qvel)
        xposafter = np.array(self.get_body_com("torso"))
        goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal_pos))  # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        #reward = goal_reward - ctrl_cost
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            task=self.get_task(),
        )

    def sample_tasks(self, num_tasks):
        a = np.array([random.random() for _ in range(num_tasks)]) * 2*np.pi
        #r = 1 * np.array([random.random() for _ in range(num_tasks)]) ** 0.5
        #a = np.array([0.75 for _ in range(num_tasks)]) * np.pi
        r = 3 * np.array([random.random() for _ in range(num_tasks)]) ** 0.5
        #self.wind = np.array([random.random() * 0.1 - 0.05,random.random() * 0.1 - 0.05])
        return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

    def set_task(self, task):
        self.goal_pos = task

    def get_task(self):
        return self.goal_pos

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

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

    def get_state(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

class SparseAntGoalEnv(AntGoalEnv):
    def __init__(self, max_episode_steps=50, goal_radius=0.2):  # , stochastic_moves=True):
        self.goal_radius = goal_radius
        # self.wind = np.array([0, 0])
        # self.wind = np.array([random.random() * 0.1 - 0.05,random.random() * 0.1 - 0.05])
        super().__init__(max_episode_steps)

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(d)
        return ob, sparse_reward, done, d

class AntGoalOracleEnv(AntGoalEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.goal_pos,
        ])
