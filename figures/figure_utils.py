#########
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

#################

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

#################

