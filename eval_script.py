import argparse
from config.dm_control import args_reacher_contrabar, args_sparse_reacher_contrabar
from config.pointrobot import args_pointrobot_image_contrabar
from config.mujoco import args_sparse_ant_goal_contrabar, args_cheetah_dir_contrabar, args_cheetah_vel_contrabar
from config.panda_gym import args_sparse_panda_reacher_contrabar, args_sparse_panda_reacher_wind_contrabar
# from config.mujoco import args_peg_insertion_image_varibad, args_walker_varibad
from config.mujoco import args_ant_dir_contrabar, args_ant_goal_contrabar, args_humanoid_dir_contrabar
from cpc import contrabarCPC
from scipy import stats
from environments.parallel_envs import make_vec_envs
from utils import evaluation
import pickle
import torch
import numpy as np
panda_augment_params = {1: {'table_color': [45/255,85/255,255/255,1],
            'plane_color': [1., 1., 159/255, 1]},
 2: {'table_color': [30/255,80/255,200/255,1],
            'plane_color': [1., 180/255., 70/255, 1]},
 3: {'table_color': [80/255,80/255,100/255,1],
            'plane_color': [1., 200/255., 50/255, 1]},
 4: {'table_color': [80/255,80/255,220/255,1],
            'plane_color': [1., 200/255., 100/255, 1]}
}
ant_goal_exps = ['contrabar_51__27:04_23:05:32/',
                 'contrabar_52__27:04_23:05:56/',
                 'contrabar_53__27:04_23:07:24/',
                 'contrabar_54__27:04_23:07:41/',
                 'contrabar_55__28:04_21:42:57/',
                 'contrabar_56__28:04_21:43:27/',
                 'contrabar_57__28:04_21:48:42/',
                 'contrabar_58__28  :04_21:48:55/',
                 'contrabar_59__29:04_21:17:59/',
                 'contrabar_60__29:04_21:18:17/']
ant_goal_gru_exps = ['contrabar_44__18:04_21:55:26/',
                     'contrabar_46__20:04_11:04:10/',
                     'contrabar_47__20:04_13:54:10/',
                     'contrabar_48__22:04_09:11:04/',
                     'contrabar_49__23:04_21:14:18/',
                     'contrabar_50__23:04_21:14:45/']
ant_goal_naive_gru_exps = ['contrabar_33__19:03_08:59:14/']
walker_exps = ['contrabar_92__07:01_07:27:35/',
        'contrabar_93__07:01_13:35:33/',
        'contrabar_94__07:01_13:36:21/',
        'contrabar_95__07:01_13:37:44/',
        'contrabar_96__07:01_13:38:59/',
        'contrabar_97__10:01_09:57:55/',
        'contrabar_98__11:01_13:05:08/',
        'contrabar_99__11:01_13:06:22/',
        'contrabar_12__11:01_13:07:13/',
        'contrabar_13__13:01_15:58:12/']
humanoid_exps = ['contrabar_17__03:01_11:29:20/',
                 'contrabar_20__05:01_23:10:35/',
                 'contrabar_21__05:01_23:10:38/',
                 'contrabar_22__05:01_23:43:18/',
                 'contrabar_23__06:01_15:47:08/',
                 'contrabar_24__07:01_12:42:22/',
                 'contrabar_25__08:01_22:20:19/',
                 'contrabar_26__09:01_20:16:22/',
                 'contrabar_27__10:01_21:02:47/',
                 'contrabar_28__10:01_21:07:54/']

reacher_cpc_exp = ['contrabar_51__17:01_19:31:07/',
                   'contrabar_52__17:01_19:41:59/',
                   'contrabar_53__17:01_19:42:59/',
                   'contrabar_54__17:01_19:43:38/',
                   'contrabar_55__17:01_19:44:25/']
panda_reacher_cpc_exp = ['contrabar_16__17:01_16:41:07/',
                         'contrabar_66__20:01_08:56:42/',
                         'contrabar_67__20:01_08:57:17/',
                         'contrabar_68__20:01_10:52:49/']
                         #'contrabar_69__20:01_09:01:36/']
panda_reacher_wind_cpc_exp = ['contrabar_22__20:01_11:08:52/',
                              'contrabar_23__20:01_11:10:34/',
                              'contrabar_24__20:01_19:56:02/',
                              'contrabar_25__23:01_09:16:19/',
                              'contrabar_26__23:01_09:17:34/']
def main(model_location=None):
    parser = argparse.ArgumentParser()
    args, rest_args = parser.parse_known_args()
    # args = args.get_args(rest_args)
    # args = args_sparse_reacher_varibad.get_args(rest_args)
    # args = args_sparse_panda_reacher_varibad.get_args(rest_args)
    # args = args_peg_insertion_image_varibad.get_args(rest_args)
    # args = args_ant_dir_varibad.get_args(rest_args)
    # args = args_cheetah_dir_varibad.get_args(rest_args)
    args = args_ant_goal_contrabar.get_args(rest_args)
    # args = args_walker_varibad.get_args(rest_args)
    # args = args_humanoid_dir_varibad.get_args(rest_args)
    # args = args_sparse_panda_reacher_wind_varibad.get_args(rest_args)


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
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_67__19:10_06:30:30//'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_70__23:10_14:29:12/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v1/contrabar_77__29:10_10:30:01/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v1/contrabar_78__30:10_14:25:47/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v1/contrabar_79__02:11_11:04:02//'
    if model_location is None:
        # model_location = '/mnt/data/erac/logs_AntGoal-v0/varibad_58__09:05_22:52:56/'
        model_location = '/mnt_/data/logs_CustomReach-v1/contrabar_69__20:01_09:01:36/'
    args.seed = 22
    with open(model_location + 'models/env_rew_rms.pkl', 'rb') as f:
        a = pickle.load(f)
    env = make_vec_envs(env_name=args.env_name,
                        seed=args.seed,
                        num_processes=1,
                        gamma=0.99,
                        device=torch.device('cpu'),
                        episodes_per_task=3,
                        normalise_rew=True, ret_rms=a,
                        rank_offset=16 + 42,  # not sure if the temp folders would otherwise clash
                        tasks=None
                        )
    args.max_trajectory_len = env._max_episode_steps
    args.max_trajectory_len *= args.max_rollouts_per_task
    args.state_dim = env.observation_space.shape
    args.action_dim = env.action_space.shape[0]
    cpc_encoder = contrabarCPC(args, None, lambda: 100)
    encoder = torch.load(model_location + 'models/encoder.pt')
    policy = torch.load(model_location + 'models/policy.pt')
    # cpc_mlp = torch.load(model_location + 'models/cpc_mlp.pt')
    # action_gru = torch.load(model_location + 'models/action_gru.pt')
    cpc_encoder.encoder = encoder
    # cpc_encoder.mlp = cpc_mlp
    # cpc_encoder.action_gru = action_gru
    args.num_processes = 1
    returns_per_episode, eval_cpc_stats, _ = evaluation.evaluate(args, policy, ret_rms=a, tasks=None, iter_idx=100,
                                                    encoder=cpc_encoder, num_episodes=5,
                                                    env_name='AntGoal-v0')
    print(returns_per_episode)
    # print(eval_cpc_stats.cpc_loss)
    return returns_per_episode
if __name__ == '__main__':
    # main()
    returns = []
    # for model_location in ['contrabar_81__21:11_15:32:14/']*10:
    # for model_location in ['contrabar_16__17:01_16:41:07/']*10:
    for model_location in ant_goal_exps:
        print(model_location)
        returns_ = []
        for _ in range(10):
            returns_.append(main('/mnt/data/erac/logs_AntGoal-v0/' + model_location).detach().cpu().numpy())
        returns.append(np.mean(returns_,axis=0))
    # print(returns)
    returns = np.concatenate(returns, axis=0)
    np.save('/mnt/data/erac/logs_AntGoal-v0/test_perf_contrabar.npy', returns)
# import numpy as np
# import matplotlib.pyplot as plt
# import pybullet
# import time
# import pybullet_data
#
# plt.ion()
#
# img = np.random.rand(200, 320)
# #img = [tandard_normal((50,100))
# image = plt.imshow(img, interpolation='none', animated=True, label="blah")
# ax = plt.gca()
#
# #pybullet.connect(pybullet.GUI)
# pybullet.connect(pybullet.DIRECT)
#
# pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
# pybullet.loadURDF("plane.urdf", [0, 0, -1])
# pybullet.loadURDF("r2d2.urdf")
#
# camTargetPos = [0, 0, 0]
# cameraUp = [0, 0, 1]
# cameraPos = [1, 1, 1]
# pybullet.setGravity(0, 0, -10)
#
# pitch = -10.0
#
# roll = 0
# upAxisIndex = 2
# camDistance = 4
# pixelWidth = 320
# pixelHeight = 200
# nearPlane = 0.01
# farPlane = 100
#
# fov = 60
#
# main_start = time.time()
# while (1):
#   for yaw in range(0, 360, 10):
#     pybullet.stepSimulation()
#     start = time.time()
#
#     viewMatrix = pybullet.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch,
#                                                             roll, upAxisIndex)
#     aspect = pixelWidth / pixelHeight
#     projectionMatrix = pybullet.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
#     img_arr = pybullet.getCameraImage(pixelWidth,
#                                       pixelHeight,
#                                       viewMatrix,
#                                       projectionMatrix,
#                                       shadow=1,
#                                       lightDirection=[1, 1, 1],
#                                       renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
#     stop = time.time()
#     print("renderImage %f" % (stop - start))
#
#     w = img_arr[0]  #width of the image, in pixels
#     h = img_arr[1]  #height of the image, in pixels
#     rgb = img_arr[2]  #color data RGB
#     dep = img_arr[3]  #depth data
#
#     print('width = %d height = %d' % (w, h))
#
#     #note that sending the data to matplotlib is really slow
#
#     #reshape is needed
#     np_img_arr = np.reshape(rgb, (h, w, 4))
#     np_img_arr = np_img_arr * (1. / 255.)
#
#     #show
#     #plt.imshow(np_img_arr,interpolation='none',extent=(0,1600,0,1200))
#     #image = plt.imshow(np_img_arr,interpolation='none',animated=True,label="blah")
#
#     image.set_data(np_img_arr)
#     ax.plot([0])
#     #plt.draw()
#     #plt.show()
#     plt.pause(0.01)
#     #image.draw()
#
# main_stop = time.time()
#
# print("Total time %f" % (main_stop - main_start))
#
# pybullet.resetSimulation()
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from scipy import stats
#
# sns.color_palette("dark")
#
# def set_default_plot_params():
#
#     plt.rcParams['font.size'] = 40
#     mpl.rcParams['ytick.labelsize'] = 15  # 21
#     mpl.rcParams['xtick.labelsize'] = 15  # 21
#     plt.rcParams["font.family"] = "Verdana"
#     plt.rcParams["font.sans-serif"] = "Verdana"
#     plt.rcParams['axes.labelsize'] = 17  # 21
#     plt.rcParams['axes.titlesize'] = 17  # 25
#     plt.rcParams['axes.linewidth'] = 0.6
#     plt.rcParams['legend.fontsize'] = 14  # 22
#     plt.rcParams["savefig.format"] = 'pdf'
#     plt.rcParams['axes.edgecolor'] = 'grey'
#     plt.rcParams['axes.edgecolor'] = 'grey'
#     plt.rcParams['axes.linewidth'] = 1
# a = np.load('test_perf_augment_contrabar_2.npy')
# d = np.load('test_perf_non_augment_contrabar_2.npy')
# b = np.load('test_perf_augment_contrabar_3.npy')
# e = np.load('test_perf_non_augment_contrabar_3.npy')
# c = np.load('test_perf_augment_contrabar_4.npy')
# f = np.load('test_perf_non_augment_contrabar_4.npy')
# augmented = np.vstack([a,b,c])
# non_augmented = np.vstack([d,e,f])
# set_default_plot_params()
# fig = plt.figure(figsize=(8, 6))
# from matplotlib import gridspec
# gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
# ax0 = plt.subplot(gs[0])
# ax1 = plt.subplot(gs[1])
# axes = [ax0, ax1]
# arr = [augmented, non_augmented]
# for j in range(2):
#     y_mean = np.mean(arr[j], axis=0)
#     y_std = np.std(arr[j], axis=0)
#     y_se = stats.sem(arr[j], axis=0)
#     y_cfi = stats.norm.interval(0.95, loc=y_mean, scale=y_se)
#     x = np.arange(1,4)
#     p = axes[0].plot(x, y_mean, marker='x', linewidth=2,
#                               label='No augmentation' if j % 2 else 'Augmentation')
#     axes[0].fill_between(x, y_cfi[0], y_cfi[1], facecolor=p[0].get_color(), alpha=0.2)
# axes[0].set_ylim([0,40])
# axes[0].set_xticks(range(1, 4))
# axes[0].set_xlabel('Episode #')
# axes[0].set_ylabel('Average return over domains')
# axes[0].set_title('Results for different domains',pad=1.6)
# axes[0].legend(loc='upper right')
# axes[0].grid()
# axes[0].set_yticks([0,5,10,15,20,25,30,35])
# e = np.load("merged.npy")
# axes[1].imshow(e, aspect='auto')
# axes[1].set_xticks([])
# axes[1].set_yticks([])
# plt.tight_layout(w_pad=0.01)

# plt.plot(np.arange(1,301)*1524, y_mean, linestyle='-', linewidth=2,label='offline ContraBAR')
# plt.show()
# plt.close()
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sns.color_palette("dark")
#sns.set(style="darkgrid")


def set_default_plot_params():

    plt.rcParams['font.size'] = 40
    mpl.rcParams['ytick.labelsize'] = 15  # 21
    mpl.rcParams['xtick.labelsize'] = 15  # 21
    plt.rcParams["font.family"] = "Verdana"
    plt.rcParams["font.sans-serif"] = "Verdana"
    plt.rcParams['axes.labelsize'] = 16  # 21
    plt.rcParams['axes.titlesize'] = 22  # 25
    plt.rcParams['axes.linewidth'] = 0.6
    plt.rcParams['legend.fontsize'] = 14  # 22
    plt.rcParams["savefig.format"] = 'pdf'
    plt.rcParams['axes.edgecolor'] = 'grey'
    plt.rcParams['axes.edgecolor'] = 'grey'
# fig = plt.figure(figsize=(6 ,6))
# xticks = np.hstack([np.linspace(0,4e5,9),4.6e5])
# plt.plot(np.arange(1,301)*1524, y_mean, linestyle='-', linewidth=2,label='offline ContraBAR')
# plt.xlabel('Steps')
# plt.ylabel('Average returns')
# plt.title('Offline Semi-Circle')
# plt.legend()
# plt.xticks(xticks)
# plt.ticklabel_format(axis='x',style='scientific',scilimits=(0,0))
# plt.grid()
# plt.savefig('/Users/erachoshen/Documents/masters/varibad_cpc/figures/offline_contrabar.png')
# plt.legend()
# plt.savefig('/Users/erachoshen/Documents/masters/varibad_cpc/figures/offline_contrabar.png')
# plt.fill_between(np.arange(1,301)*1524, y_cfi[0], y_cfi[1],alpha=0.2)
# plt.savefig('/Users/erachoshen/Documents/masters/varibad_cpc/figures/offline_contrabar.png')

# plt.rcParams["savefig.format"] = 'pdf'
# fig, axes = plt.subplots(2,2,figsize=(6,4))
# axes.flatten()[0].imshow(frames[1])
# axes.flatten()[1].imshow(frames[17])
# axes.flatten()[2].imshow(frames[19])
# axes.flatten()[3].imshow(frames[63])
# for ax in axes.flatten():
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.tight_layout(w_pad=0.01, h_pad=0.01)

