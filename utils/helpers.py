import os
import pickle
# import pickle5 as pickle
import random
import warnings
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from environments.parallel_envs import make_vec_envs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def save_models(args, logger, policy, vae, envs, iter_idx):
#     # TODO: save parameters, not entire model
#
#     save_path = os.path.join(logger.full_output_folder, 'models')
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#     try:
#         torch.save(policy.actor_critic, os.path.join(save_path, "policy{0}.pt".format(iter_idx)))
#     except AttributeError:
#         torch.save(policy.policy, os.path.join(save_path, "policy{0}.pt".format(iter_idx)))
#     torch.save(vae.encoder, os.path.join(save_path, "encoder{0}.pt".format(iter_idx)))
#     if vae.state_decoder is not None:
#         torch.save(vae.state_decoder, os.path.join(save_path, "state_decoder{0}.pt".format(iter_idx)))
#     if vae.reward_decoder is not None:
#         torch.save(vae.reward_decoder,
#                    os.path.join(save_path, "reward_decoder{0}.pt".format(iter_idx)))
#     if vae.task_decoder is not None:
#         torch.save(vae.task_decoder, os.path.join(save_path, "task_decoder{0}.pt".format(iter_idx)))
#
#     # save normalisation params of envs
#     if args.norm_rew_for_policy:
#         rew_rms = envs.venv.ret_rms
#         save_obj(rew_rms, save_path, "env_rew_rms{0}.pkl".format(iter_idx))
#     if args.norm_obs_for_policy:
#         obs_rms = envs.venv.obs_rms
#         save_obj(obs_rms, save_path, "env_obs_rms{0}.pkl".format(iter_idx))


def reset_env(env, args, indices=None, state=None, **kwargs):
    """ env can be many environments or just one """
    # reset all environments
    if (indices is None) or (len(indices) == args.num_processes):
        state = env.reset(**kwargs).float().to(device)
    # reset only the ones given by indices
    else:
        assert state is not None
        for i in indices:
            state[i] = env.reset(index=i)

    task = torch.from_numpy(env.get_task()).float().to(device)# if args.pass_task_to_policy else None
        
    return state, task


def squash_action(action, args):
    if args.norm_actions_post_sampling:
        return torch.tanh(action)
    else:
        return action


def env_step(env, action, args):
    act = squash_action(action.detach(), args)
    next_obs, reward, done, infos = env.step(act)

    if isinstance(next_obs, list):
        next_obs = [o.to(device) for o in next_obs]
    else:
        next_obs = next_obs.to(device)
    if isinstance(reward, list):
        reward = [r.to(device) for r in reward]
    else:
        reward = reward.to(device)

    task = torch.from_numpy(env.get_task()).float().to(device)# if (args.pass_task_to_policy or args.decode_task) else None

    return [next_obs, task], reward, done, infos


def select_action_cpc(args, policy, deterministic, hidden_latent, state=None, task=None):
    """ Select action using the policy. """
    action = policy.act(state=state, latent=hidden_latent, task=task, deterministic=deterministic)
    if isinstance(action, list) or isinstance(action, tuple):
        value, action = action
    else:
        value = None
    action = action.to(device)
    return value, action


def update_encoding_cpc(encoder, next_obs, action, reward, done, hidden_state):
    # reset hidden state of the recurrent net when we reset the task
    if done is not None:
        hidden_state = encoder.reset_hidden(hidden_state, done)

    with torch.no_grad():
        hidden_state = encoder(actions=action.float(),
                                                                          states=next_obs,
                                                                          rewards=reward,
                                                                          hidden_state=hidden_state,
                                                                          return_prior=False)

    # TODO: move the sampling out of the encoder!

    return hidden_state

def seed(seed, deterministic_execution=False):
    print('Seeding random, torch, numpy.')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    if deterministic_execution:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print('Note that due to parallel processing results will be similar but not identical. '
              'Use only one process and set --deterministic_execution to True if you want identical results '
              '(only recommended for debugging).')


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




class FeatureExtractor(nn.Module):
    """ Used for extrating features for states/actions/rewards """

    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc_1 = nn.Linear(input_size, 64)
            self.fc_2 = nn.Linear(64, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            output = self.activation_function(self.fc_1(inputs))
            output = self.fc_2(output)
            return output
        else:
            return torch.zeros(0, ).to(device)


def sample_gaussian(mu, logvar, num=None):
    std = torch.exp(0.5 * logvar)
    if num is not None:
        std = std.repeat(num, 1)
        mu = mu.repeat(num, 1)
    eps = torch.randn_like(std)
    return mu + std * eps


def save_obj(obj, folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'rb') as f:
        return pickle.load(f)


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # PyTorch version.
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape).float().to(device)
        self.var = torch.ones(shape).float().to(device)
        self.count = epsilon

    def update(self, x):
        x = x.view((-1, x.shape[-1]))
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))


def get_task_dim(args):
    env = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                        gamma=args.policy_gamma, device=device,
                        episodes_per_task=args.max_rollouts_per_task,
                        normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                        tasks=None
                        )
    return env.task_dim


def get_num_tasks(args):
    env = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                        gamma=args.policy_gamma, device=device,
                        episodes_per_task=args.max_rollouts_per_task,
                        normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                        tasks=None
                        )
    try:
        num_tasks = env.num_tasks
    except AttributeError:
        num_tasks = None
    return num_tasks


def clip(value, low, high):
    """Imitates `{np,tf}.clip`.

    `torch.clamp` doesn't support tensor valued low/high so this provides the
    clip functionality.

    TODO(hartikainen): The broadcasting hasn't been extensively tested yet,
        but works for the regular cases where
        `value.shape == low.shape == high.shape` or when `{low,high}.shape == ()`.
    """
    low, high = torch.tensor(low), torch.tensor(high)

    assert torch.all(low <= high), (low, high)

    clipped_value = torch.max(torch.min(value, high), low)
    return clipped_value

def get_pos_grid(pos, device):
    X_SIZE = 5
    Y_SIZE = 5
    grid = torch.zeros(len(pos), X_SIZE*Y_SIZE, dtype=torch.float32, device=device)
    idx = torch.tensor(np.array([pos_[0]*(Y_SIZE) + pos_[1] for pos_ in pos]),device=device).view(-1, 1)
    grid.scatter_(dim=1, index=idx, src=torch.full_like(grid, fill_value=1))
    return grid.view(-1, X_SIZE, Y_SIZE)

def rgb2gray(rgb):
    gray = 0.2989*rgb[...,0]+0.587*rgb[...,1]+0.114*rgb[...,2]
    return np.expand_dims(gray, axis=0)
