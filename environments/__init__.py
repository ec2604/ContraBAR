from gym.envs.registration import register

# Mujoco
# ----------------------------------------

# - randomised reward functions

register(
    'AntDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntDir2D-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDir2DEnv',
            'max_episode_steps': 200},
    max_episode_steps=200,
)

register(
    'AntGoal-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal:AntGoalEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)
register(
    'AntGoalImage-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal_image:AntGoalEnvImage',
            'max_episode_steps': 200},
    max_episode_steps=200
)
register(
    'SparseAntGoalImage-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal_image:SparseAntGoalEnvImage',
            'max_episode_steps': 50},
    max_episode_steps=50
)


register(
    'HalfCheetahDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_dir:HalfCheetahDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahVel-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel:HalfCheetahVelEnv',
            'max_episode_steps': 50},
    max_episode_steps=50
)
register(
    'HalfCheetahVelWind-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel:HalfCheetahVelWindEnv',
            'max_episode_steps': 50},
    max_episode_steps=50
)
register(
    'HalfCheetahVelImage-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel_image:HalfCheetahVelEnvImage',
            'max_episode_steps': 50},
    max_episode_steps=50
)

register(
    'HumanoidDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.humanoid_dir:HumanoidDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

# - randomised dynamics

register(
    id='Walker2DRandParams-v0',
    entry_point='environments.mujoco.rand_param_envs.walker2d_rand_params:Walker2DRandParamsEnv',
    max_episode_steps=200
)

register(
    id='HopperRandParams-v0',
    entry_point='environments.mujoco.rand_param_envs.hopper_rand_params:HopperRandParamsEnv',
    max_episode_steps=200
)


# # 2D Navigation
# # ----------------------------------------
#
register(
    'PointEnv-v0',
    entry_point='environments.navigation.point_robot:PointEnv',
    kwargs={
            'max_episode_steps': 100,
            'goal_sampler': 'semi-circle'
            },
    max_episode_steps=100,
)
register(
    'PointEnvImage-v0',
    entry_point='environments.navigation.point_robot_image:PointEnv',
    kwargs={
            'max_episode_steps': 100,
            'goal_sampler': 'semi-circle'
            },
    max_episode_steps=100,
)
register(
    'SparsePointEnvImage-v0',
    entry_point='environments.navigation.point_robot_image:SparsePointEnv',
    kwargs={
            'max_episode_steps': 100,
            'goal_sampler': 'semi-circle'
            },
    max_episode_steps=100,
)
register(
    'PointEnvWind-v0',
    entry_point='environments.navigation.point_robot_wind:PointEnvWind',
    kwargs={'max_episode_steps': 100,
            'goal_sampler': 'semi-circle'
            },
    max_episode_steps=100,
)
register(
    'SparsePointEnv-v0',
    entry_point='environments.navigation.point_robot:SparsePointEnv',
    kwargs={'goal_radius': 0.2,
            'max_episode_steps': 100,
            'goal_sampler': 'semi-circle'
            },
    max_episode_steps=100,
)
register(
    'SparsePointEnvWind-v0',
    entry_point='environments.navigation.point_robot_wind:SparsePointEnvWind',
    kwargs={'goal_radius': 0.2,
            'max_episode_steps': 100,
            'goal_sampler': 'semi-circle'
            },
    max_episode_steps=100,
)
#
# # GridWorld
# # ----------------------------------------

register(
    'GridNavi-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells': 5, 'num_steps': 15},
)

#
# # Reacher
# # ------------------------------------------
register(
    'Reacher-v0',
    entry_point='environments.dm_control.reacher:ReacherEnv',
    kwargs={'max_episode_steps': 100, 'image_size': 64, 'action_repeat': 4, 'dense': True},
    max_episode_steps=100
)

register(
    'SparseReacher-v0',
    entry_point='environments.dm_control.reacher:ReacherEnv',
    kwargs={'max_episode_steps': 50, 'image_size': 64, 'action_repeat': 4, 'dense': False},
    max_episode_steps=50
)
#
# # Procgen
# # ------------------------------------------
register(
    'Maze-v0',
    entry_point='environments.procgen.maze:MazeEnv',
    kwargs={'use_backgrounds': False, 'distribution_mode': 'memory',
                            'render_mode': 'rgb_array', 'num_levels': 1},
    max_episode_steps=100
)