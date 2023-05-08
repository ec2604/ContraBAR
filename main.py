"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters from the respective config file.
"""
import argparse
import warnings
import wandb
import numpy as np
import torch

# get configs
from config.gridworld import  args_grid_contrabar
from config.pointrobot import args_pointrobot_contrabar, args_pointrobot_wind_contrabar, \
    args_pointrobot_wind_dense_contrabar, args_pointrobot_image_contrabar
from config.mujoco import args_cheetah_dir_contrabar, args_cheetah_vel_contrabar, \
    args_cheetah_vel_image_contrabar, args_ant_dir_contrabar, args_ant_goal_contrabar, \
    args_walker_contrabar, args_humanoid_dir_contrabar, args_ant_goal_image_contrabar, \
    args_sparse_ant_goal_image_contrabar, args_sparse_ant_goal_contrabar, args_offline_sparse_ant_goal_image_contrabar, \
    args_peg_insertion_image_contrabar
from config.dm_control import args_reacher_contrabar, args_sparse_reacher_contrabar, args_sparse_proprio_reacher_contrabar
from config.panda_gym import args_sparse_panda_reacher_contrabar, args_sparse_panda_reacher_wind_contrabar
from environments.parallel_envs import make_vec_envs
from learner import Learner
from metalearner_cpc import MetaLearner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='gridworld_contrabar')
    args, rest_args = parser.parse_known_args()
    env = args.env_type

    # --- GridWorld ---

    if env == 'gridworld_contrabar':
        args = args_grid_contrabar.get_args(rest_args)

    # --- PointRobot 2D Navigation ---

    elif env == 'pointrobot_contrabar':
        args = args_pointrobot_contrabar.get_args(rest_args)
    elif env == 'pointrobot_wind_contrabar':
        args = args_pointrobot_wind_contrabar.get_args(rest_args)
    elif env == 'pointrobot_wind_dense_contrabar':
        args = args_pointrobot_wind_dense_contrabar.get_args(rest_args)
    elif env =='pointrobot_image_contrabar':
        args = args_pointrobot_image_contrabar.get_args(rest_args)
    # --- MUJOCO ---

    # - PegInsertion -
    elif env == 'peg_insertion_image':
        args = args_peg_insertion_image_contrabar.get_args(rest_args)
    # - CheetahDir -
    elif env == 'cheetah_dir_contrabar':
        args = args_cheetah_dir_contrabar.get_args(rest_args)
    #
    # - CheetahVel -
    elif env == 'cheetah_vel_contrabar':
        args = args_cheetah_vel_contrabar.get_args(rest_args)
    elif env == 'cheetah_vel_image_contrabar':
        args = args_cheetah_vel_image_contrabar.get_args(rest_args)
    #
    # - AntDir -
    elif env == 'ant_dir_contrabar':
        args = args_ant_dir_contrabar.get_args(rest_args)
    #
    # - AntGoal -
    elif env == 'ant_goal_contrabar':
        args = args_ant_goal_contrabar.get_args(rest_args)
    elif env == 'sparse_ant_goal_contrabar':
        args = args_sparse_ant_goal_contrabar.get_args(rest_args)
    elif env == 'offline_sparse_ant_goal_contrabar':
        args = args_offline_sparse_ant_goal_image_contrabar.get_args(rest_args)
    elif env == 'ant_goal_image_contrabar':
        args = args_ant_goal_image_contrabar.get_args(rest_args)
    elif env == 'sparse_ant_goal_image_contrabar':
        args = args_sparse_ant_goal_image_contrabar.get_args(rest_args)
    #
    # - Walker -
    elif env == 'walker_contrabar':
        args = args_walker_contrabar.get_args(rest_args)
    #
    # - HumanoidDir -
    elif env == 'humanoid_dir_contrabar':
        args = args_humanoid_dir_contrabar.get_args(rest_args)
    #
    # - Reacher -
    elif env == 'reacher':
        args = args_reacher_contrabar.get_args(rest_args)
    elif env == 'sparse_reacher':
        args = args_sparse_reacher_contrabar.get_args(rest_args)
    elif env == 'sparse_proprio_reacher':
        args = args_sparse_proprio_reacher_contrabar.get_args(rest_args)
    # - PandaGym
    elif env == 'panda_reacher':
        args = args_sparse_panda_reacher_contrabar.get_args(rest_args)
    elif env == 'panda_reacher_wind':
        args = args_sparse_panda_reacher_wind_contrabar.get_args(rest_args)
    else:
        raise Exception("Invalid Environment")


    # warning for deterministic execution
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    # if we're normalising the actions, we have to make sure that the env expects actions within [-1, 1]
    if args.norm_actions_pre_sampling or args.norm_actions_post_sampling:
        envs = make_vec_envs(env_name=args.env_name, seed=0, num_processes=args.num_processes,
                             gamma=args.policy_gamma, device='cpu',
                             episodes_per_task=args.max_rollouts_per_task,
                             normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                             tasks=None,
                             )
        assert np.unique(envs.action_space.low) == [-1]
        assert np.unique(envs.action_space.high) == [1]



    # begin training (loop through all passed seeds)
    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed
    for seed in seed_list:
        print('training', seed)
        args.seed = seed
        args.action_space = None

        if args.disable_metalearner:
            # If `disable_metalearner` is true, the file `learner.py` will be used instead of `metalearner.py`.
            # This is a stripped down version without encoder, decoder, stochastic latent variables, etc.
            learner = Learner(args)
        else:
            learner = MetaLearner(args)
        learner.train()


if __name__ == '__main__':
    main()
