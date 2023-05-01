import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette("dark")

def set_default_plot_params():

    plt.rcParams['font.size'] = 40
    mpl.rcParams['ytick.labelsize'] = 21 #21
    mpl.rcParams['xtick.labelsize'] = 21 #21
    plt.rcParams["font.family"] = "Verdana"
    plt.rcParams["font.sans-serif"] = "Verdana"
    plt.rcParams['axes.labelsize'] = 21 #21
    plt.rcParams['axes.titlesize'] = 25 #25
    plt.rcParams['axes.linewidth'] = 0.6
    plt.rcParams['legend.fontsize'] = 14 #22
    plt.rcParams["savefig.format"] = 'pdf'
    plt.rcParams['axes.edgecolor'] = 'grey'
    plt.rcParams['axes.edgecolor'] = 'grey'
    plt.rcParams['axes.linewidth'] = 1


import numpy as np

from scipy import stats


methods = ['varibad', 'cpc', 'recurrent']
# methods = ['cpc', 'rl2']

# methods = ['cpc']
my_envs = {'Cheetah-Dir': 'cheetah_dir',
           'Ant-Dir': 'ant_dir',
           'Cheetah-Vel': 'cheetah_vel',
           'Humanoid-Dir': 'humanoid',
           'Walker': 'walker',
           'Ant-Goal': 'ant_goal'}
# my_envs = {'Ant-Goal': 'ant_goal_no_gru'}
# my_envs = {'Reacher': 'reacher',
#            'Panda Reacher': 'custom_reacher',
#            'Panda Wind': 'custom_wind_reacher',
# }

my_colors = {
    'varibad': sns.color_palette("bright", 10)[8],
    # 'rl2': sns.color_palette("deep", 10)[6],
    'cpc': sns.color_palette("bright", 10)[3],
    'recurrent': sns.color_palette("bright", 10)[5]
}
my_labels = {
    'varibad': 'VariBAD',
    'cpc': 'ContraBAR',
    'rl2': '$RL^2$',
    'recurrent': 'RMF'

}

my_linestyles = {
    'varibad': '-',
    'cpc': '--',
    'rl2': '-',
    'recurrent': '-.'
    }
my_files = {
    'varibad': 'varibad',
    'cpc': 'contrabar',
    # 'rl2': 'rl2',
    'recurrent': 'recurrent'
}

# plot results for each env and method
set_default_plot_params()
# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# ax = np.array([ax])
fig, ax = plt.subplots(2, 3, figsize=(15,8))
# fig, ax = plt.subplots(1, 3, figsize=(15,6))

for i, (env_key, env_val) in enumerate(my_envs.items()):
    for method in methods:
        print(method)

        # read in x-values (steps)
        # read in y-values (returns)
        try:
            y = np.load(f'/home/erac/varibad_cpc/end_performance_per_episode/{env_val}_' + my_files[method] + '.npy')
        except:
            print('fail')
            continue
        x = np.arange(1, y.shape[1]+1).astype(np.int)

        # compute averages and confidence intervals across seeds
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        y_se = stats.sem(y, axis=0)
        y_cfi = stats.norm.interval(0.95, loc=y_mean, scale=y_se)
        p = ax.reshape(-1)[i].plot(x, y_mean, linestyle=my_linestyles[method], marker='x', linewidth=2, label=my_labels[method],
                     color=my_colors[method])
        ax.reshape(-1)[i].fill_between(x, y_cfi[0], y_cfi[1], facecolor=p[0].get_color(), alpha=0.2)
    ax.reshape(-1)[i].set_title(f'{env_key}')
    ax.reshape(-1)[i].set_xticks(range(1, 6))
    ax.reshape(-1)[i].grid()
    img = False
    if img:
        if i == 0:
            ax.reshape(-1)[i].set_ylim([0, 50])
            ax.reshape(-1)[i].set_ylabel('Avg Return')
            ax.reshape(-1)[i].legend(loc='lower right')
        if i == 1:
            ax.reshape(-1)[i].set_ylim([0, 50])
        if i == 2:
            ax.reshape(-1)[i].set_ylim([40, 50])
        if i == 1:
            ax.reshape(-1)[i].set_xlabel('Episodes')
    else:
        if i == 0:
            ax.reshape(-1)[i].set_ylim([1200, 2200])
            ax.reshape(-1)[i].legend(loc='lower right')
        if i == 0 or i == 3:
            ax.reshape(-1)[i].set_ylabel('Avg Return')
        if i == 2:
            ax.reshape(-1)[i].set_ylim([-60, 0])
        if i == 4:
            ax.reshape(-1)[i].set_ylim([900, 1300])
            ax.reshape(-1)[i].set_xlabel('Episodes')


# plt.show()
# plt.tight_layout(h_pad=0.3, w_pad=0.3)
plt.tight_layout(w_pad=0.5)
plt.savefig('./test_perf')