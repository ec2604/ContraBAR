import argparse
from config.dm_control import args_reacher_contrabar, args_sparse_reacher_contrabar
from config.pointrobot import args_pointrobot_contrabar
from config.pointrobot import args_pointrobot_image_contrabar
from config.mujoco import args_sparse_ant_goal_contrabar
from config.panda_gym import args_sparse_panda_reacher_contrabar, args_sparse_panda_reacher_wind_contrabar
from environments.parallel_envs import make_vec_envs,make_env
from tqdm import tqdm
import utils.helpers as utl
import pickle
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args, rest_args = parser.parse_known_args()
    # args = args_sparse_panda_reacher_varibad.get_args(rest_args)
    # args = args_sparse_panda_reacher_wind_varibad.get_args(rest_args)
    args = args_pointrobot_contrabar.get_args(rest_args)
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_90__16:09_21:25:28/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_16__23:09_20:19:52/'
    # model_location = '/mnt/data/erac/logs_CustomReachWind-v0/contrabar_88__16:11_11:03:02/'
    with open(model_location + 'models/env_rew_rms.pkl', 'rb') as f:
        a = pickle.load(f)
    env = make_vec_envs(env_name='CustomReachWind-v0',
                        seed=12 * 42 +10000,
                        num_processes=16,
                        gamma=0.99,
                        device=device,
                        episodes_per_task=3,
                        normalise_rew=True, ret_rms=a,
                        rank_offset=16 + 42,  # not sure if the temp folders would otherwise clash
                        tasks=None
                        )
    
    encoder = torch.load(model_location + 'models/encoder.pt')
    policy = torch.load(model_location + 'models/policy.pt')
    num_trajs = 1000
    traj_len = 150
    hidden_size = 50
    task_size = 6
    obs_shape = (5, 84, 84)
    pos_size = 3
    hidden_buffer = torch.zeros(num_trajs, traj_len, hidden_size)
    task_buffer = torch.zeros(num_trajs, task_size)
    prev_pos_buffer = np.zeros((num_trajs, traj_len, pos_size),dtype=np.float)
    prev_obs_buffer = torch.zeros(num_trajs, traj_len, *obs_shape)
    for i in tqdm(range(num_trajs // 16)):
        if i > 0:
            del episode_hidden_states
            del episode_prev_obs
            del episode_next_obs
            del episode_actions
            del episode_rewards
            del episode_returns
            del episode_pos
            del task
        episode_hidden_states, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
        episode_returns, episode_pos, task = utl.collect_data(env, args, policy, encoder)
        hidden_buffer[(i*16):(i+1)*16, ...] = episode_hidden_states.detach().cpu()
        task_buffer[(i*16):(i+1)*16, ...] = task.detach().cpu()
        prev_pos_buffer[(i*16):(i+1)*16, ...] = episode_pos.copy()
        prev_obs_buffer[(i*16):(i+1)*16, ...] = episode_prev_obs.detach().cpu()
    np.savez("/mnt/data/erac/logs_CustomReachWind-v0/panda_wind_belief_ds_seed_88.npz",
             hidden_buffer=hidden_buffer.numpy(), task_buffer=task_buffer.numpy(),
             prev_pos_buffer=prev_pos_buffer, prev_obs_buffer=prev_obs_buffer.numpy())