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
            'max_episode_steps': 100, 'goal_radius': 0.3},
    max_episode_steps=100
)
register(
    'PegInsertion-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.peg_insertion:PegSimEnv',
            'max_episode_steps': 80},
    max_episode_steps=80
)
register(
    'PegInsertion-v1',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.peg_insertion:PegSimEnvAngle',
            'max_episode_steps': 80, 'distance_threshold': 0.05},
    max_episode_steps=80
)
register(
    'PegInsertion-v2',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.peg_insertion:PegSimEnvProprio',
            'max_episode_steps': 80, 'distance_threshold': 0.05},
    max_episode_steps=80
)
register(
    'CustomReach-v0',
    entry_point='environments.panda_gym.custom_reach:CustomReachEnv',
    kwargs={'render': False},
    max_episode_steps=50
)
register(
    'CustomReach-v1',
    entry_point='environments.panda_gym.custom_reach:CustomReachEnv',
    kwargs={'render': False, 'table_color': [45/255,85/255,255/255,1],
            'plane_color': [1., 1., 159/255, 1]},
    max_episode_steps=50
)

register(
    'CustomReach-v2',
    entry_point='environments.panda_gym.custom_reach:CustomReachEnv',
    kwargs={'render': False, 'table_color': [30/255,80/255,200/255,1],
            'plane_color': [1., 180/255., 70/255, 1]},
    max_episode_steps=50
)
register(
    'CustomReach-v3',
    entry_point='environments.panda_gym.custom_reach:CustomReachEnv',
    kwargs={'render': False, 'table_color': [80/255,80/255,100/255,1],
            'plane_color': [1., 200/255., 50/255, 1]},
    max_episode_steps=50
)
register(
    'CustomReach-v4',
    entry_point='environments.panda_gym.custom_reach:CustomReachEnv',
    kwargs={'render': False, 'table_color': [80/255,80/255,220/255,1],
            'plane_color': [1., 200/255., 100/255, 1]},
    max_episode_steps=50
)
register(
    'CustomReachWind-v0',
    entry_point='environments.panda_gym.custom_reach:CustomReachWindEnv',
    kwargs={'render': False, 'table_color': [215/255,181/255,18/255,1],
            'plane_color': [102/255., 1., 159/255, 1]},
    max_episode_steps=50
)
register(
    'CustomReachWind-v1',
    entry_point='environments.panda_gym.custom_reach:CustomReachWindEnv',
    kwargs={'render': False, 'table_color': [45/255,85/255,255/255,1],
            'plane_color': [1., 1., 159/255, 1]},
    max_episode_steps=50
)

register(
    'SparseAntGoalImage-v1',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal_image:SparseAntGoalEnvImage',
            'max_episode_steps': 50, 'goal_radius': 0.3},
    max_episode_steps=50
)
register(
    'OfflineSparseAntGoalImage-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal_image:OfflineSparseAntGoalEnvImage',
            'max_episode_steps': 100, 'goal_radius': 0.3},
    max_episode_steps=100
)
register(
    'SparseAntGoal-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal:SparseAntGoalEnv',
            'max_episode_steps': 50, 'goal_radius': 0.3},
    max_episode_steps=50
)
register(
    'SparseAntGoal-v1',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal:SparseAntGoalEnv',
            'max_episode_steps': 100, 'goal_radius': 0.3},
    max_episode_steps=100
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
            'max_episode_steps': 200},
    max_episode_steps=200
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
    kwargs={'max_episode_steps': 50,
            'goal_sampler': 'constant'
            },
    max_episode_steps=50,
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
register(
    'SparseReacher-v1',
    entry_point='environments.dm_control.reacher:ReacherEnv',
    kwargs={'max_episode_steps': 10, 'image_size': 64, 'action_repeat': 4, 'dense': False},
    max_episode_steps=10
)
register(
    'SparseReacher-v2',
    entry_point='environments.dm_control.reacher:ReacherEnv',
    kwargs={'max_episode_steps': 8, 'image_size': 64, 'action_repeat': 4, 'dense': False},
    max_episode_steps=8
)
register(
    'SparseProprioReacher-v0',
    entry_point='environments.dm_control.reacher_proprio:ReacherEnv',
    kwargs={'max_episode_steps': 50, 'action_repeat': 4, 'dense': False},
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