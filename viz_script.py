import argparse
from config.dm_control import args_reacher_varibad, args_sparse_reacher_varibad
from config.pointrobot import args_pointrobot_image_varibad, args_pointrobot_varibad
from config.mujoco import args_sparse_ant_goal_varibad
from config.mujoco import args_peg_insertion_image_varibad
from config.gridworld import args_grid_varibad
from config.pointrobot import args_pointrobot_varibad
from config.panda_gym import args_sparse_panda_reacher_varibad, args_sparse_panda_reacher_wind_varibad
from environments.parallel_envs import make_vec_envs
import pickle
import torch
import numpy as np
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args, rest_args = parser.parse_known_args()
    # args = args_sparse_panda_reacher_varibad.get_args(rest_args)
    # args = args_sparse_panda_reacher_wind_varibad.get_args(rest_args)
    # args = args_peg_insertion_image_varibad.get_args(rest_args)
    # args = args_pointrobot_varibad.get_args(rest_args)
    # args = args_grid_varibad.get_args(rest_args)
    args = args_pointrobot_varibad.get_args(rest_args)

    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_90__16:09_21:25:28/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_16__23:09_20:19:52/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_21__28:09_20:57:43/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_22__29:09_08:42:54/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_28__03:10_10:28:13/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_32__06:10_10:34:02/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_36__07:10_20:29:13/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_30__05:10_16:42:57/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_38__10:10_09:41:21/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_41__10:10_14:36:02/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_46__11:10_12:31:23/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_48__11:10_21:58:14/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_43__11:10_06:48:05/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_52__14:10_23:03:52/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_42__10:10_14:36:20/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_58__17:10_13:01:30/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_70__23:10_14:29:12/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v1/contrabar_77__29:10_10:30:01/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v1/contrabar_78__30:10_14:25:47/'
    # model_location = '/mnt/data/erac/logs_CustomReachWind-v0/contrabar_88__16:11_11:03:02/'
    # model_location = '/mnt/data/erac/logs_PegInsertion-v0/contrabar_21__24:11_09:32:44/'
    # model_location = '/mnt/data/erac/logs_PegInsertion-v0/contrabar_23__28:11_15:09:37/'
    # model_location = '/mnt/data/erac/logs_PegInsertion-v1/contrabar_71__08:01_23:14:33/'
    # model_location = '/mnt/data/erac/logs_PegInsertion-v2/contrabar_40__28:12_23:24:18/'
    # model_location = '/mnt/data/erac/logs_SparsePointEnv-v0/contrabar_70__22:10_15:31:47/'
    model_location = '/mnt/data/erac/logs_SparsePointEnv-v0/contrabar_26__17:03_08:26:36/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v1/contrabar_80__21:11_12:20:40/'
    # model_location = '/mnt/data/erac/logs_GridNavi-v0/contrabar_77__14:11_13:43:44/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v1/contrabar_81__21:11_15:32:14/'


    with open(model_location + 'models/env_rew_rms.pkl', 'rb') as f:
        a = pickle.load(f)
    env = make_vec_envs(env_name='SparsePointEnv-v0',
                        seed=12 * 42 +10004,
                        num_processes=1,
                        gamma=0.99,
                        device=torch.device('cpu'),
                        episodes_per_task=3,
                        normalise_rew=True, ret_rms=a,
                        rank_offset=16 + 42,  # not sure if the temp folders would otherwise clash
                        tasks=None
                        )
    encoder = torch.load(model_location + 'models/encoder.pt')
    policy = torch.load(model_location + 'models/policy.pt')
    # belief_evaluator = torch.load("/mnt/data/erac/logs_PegInsertion-v0/belief_panda_seed_88.pt")
    unwrapped_env = env.venv.unwrapped.envs[0]
    #goal_pos = np.array([(np.pi)/2, 0.2])
    #start_pos = np.array([0.])
    #unwrapped_env.visualise_behaviour_for_script(env, args,policy, 100, encoder, image_folder='/mnt/data/erac/logs_SparseReacher-v0/test',goal_pos=goal_pos,start_pos=start_pos,belief_evaluator=None)
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy import stats

    sns.color_palette("dark")

    def set_default_plot_params():

        # plt.rcParams['font.size'] = 40
        mpl.rcParams['ytick.labelsize'] = 30  # 21
        mpl.rcParams['xtick.labelsize'] = 30  # 21
        # plt.rcParams["font.family"] = "Verdana"
        # plt.rcParams["font.sans-serif"] = "Verdana"
        plt.rcParams['axes.labelsize'] = 30  # 21
        plt.rcParams['axes.titlesize'] = 30  # 25
        # plt.rcParams['axes.linewidth'] = 0.6
        plt.rcParams['legend.fontsize'] = 30  # 22
        plt.rcParams["savefig.format"] = 'pdf'
        # plt.rcParams['axes.edgecolor'] = 'grey'
        # plt.rcParams['axes.edgecolor'] = 'grey'
        # plt.rcParams['axes.linewidth'] = 1

    set_default_plot_params()
    episode_hidden_states, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
    episode_returns = unwrapped_env.visualise_behaviour(env, args, policy, 104, encoder,
                                                        image_folder='/mnt/data/erac/logs_SparsePointEnv-v0/test',
                                                        # image_folder='/mnt/data/erac/logs_GridNavi-v0/test',
                                                        save_video=True)#,belief_evaluator=belief_evaluator)

