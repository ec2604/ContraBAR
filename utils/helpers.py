import os
import pickle
import random
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from environments.parallel_envs import make_vec_envs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    task = torch.from_numpy(env.get_task()).float().to(device)  # if args.pass_task_to_policy else None

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

    task = torch.from_numpy(env.get_task()).float().to(
        device)  # if (args.pass_task_to_policy or args.decode_task) else None

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

    def __init__(self, input_size, output_size, activation_function, same=False, mid=None):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        self.same = same
        self.mid = mid
        if self.output_size != 0:
            if not self.same:
                if mid is None:
                    self.fc_1 = nn.Linear(input_size, output_size // 2)
                    self.fc_2 = nn.Linear(output_size // 2, output_size)
                else:
                    self.fc_1 = nn.Linear(input_size, mid)
                    self.fc_2 = nn.Linear(mid, output_size)
            else:
                self.layer = nn.Identity()
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            if hasattr(self, 'mid')  and self.mid is not None:
                output = self.activation_function(self.fc_1(inputs))
                output = self.fc_2(output)
                return output
            if not self.same:
                output = self.activation_function(self.fc_1(inputs))
                output = self.fc_2(output)
                return output
            else:
                return self.layer(inputs)
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
    env = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=1,
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


def rgb2gray(rgb):
    gray = 0.2989 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.expand_dims(gray, axis=0)


def plot_grad_flow(named_parameters, figure_name, logger, iter_idx):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    fig = plt.figure()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                plt.Line2D([0], [0], color="b", lw=4),
                plt.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    logger.add_figure(figure_name, fig, iter_idx)





class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self):
        super().__init__()

    def forward(self, prediction_matrix):
        return info_nce(prediction_matrix)


def info_nce(prediction_matrix):
    max_val = prediction_matrix.max(dim=1, keepdim=True)[0].detach()
    numerator = torch.exp(prediction_matrix[:, 0:1] - max_val)
    denominator = torch.sum(torch.exp(prediction_matrix[:, 1:] - max_val), dim=1, keepdim=True) + numerator
    log_exp = -torch.log((numerator / denominator) + torch.finfo(numerator.dtype).tiny)
    return log_exp


def generate_predictor_input(hidden_states, tasks, reward_radius=0.05, train=False):
    l = 50
    z_ = 0.15 / 2
    n = 10
    if train:
        r = 0.15
        theta = np.linspace(-np.pi, np.pi, l)
        x_ = np.cos(theta) * r
        y_ = np.sin(theta) * r
        all_states = (torch.normal(0, 0.05 / 3, (n, 1, 3)).numpy() + tasks.unsqueeze(0).cpu().numpy()).reshape(-1, 3)
        r = 0.3
        x_ = y_ = np.linspace(-r, r, l)
        x, y, z = np.meshgrid(x_, y_, z_)
        square_states = np.dstack((x, y, z)).reshape(-1, 3)
        idx = np.arange(len(square_states))
        np.random.shuffle(idx)
        all_states = np.vstack([all_states, square_states[idx[:(n * l) // 5]]])

    else:
        r = 0.3
        x_ = y_ = np.linspace(-r, r, l)
        x, y, z = np.meshgrid(x_, y_, z_)
        all_states = np.dstack((x, y, z)).reshape(-1, 3)

    repeated_tasks = tasks.unsqueeze(1).repeat(1, all_states.shape[0], 1).detach().cpu().numpy()
    repeated_states = np.tile(np.expand_dims(all_states, 0), [tasks.shape[0], 1, 1])
    labels = torch.tensor(np.linalg.norm(repeated_tasks - repeated_states, axis=-1) <= reward_radius).to(torch.float)
    repeated_labels = labels.unsqueeze(0).repeat(hidden_states.shape[0], 1, 1).unsqueeze(-1).to(device)
    repeated_hidden = torch.tensor(hidden_states.unsqueeze(-2).repeat(1, 1, all_states.shape[0], 1)).cpu()
    repeated_states_for_hidden = torch.tensor(
        np.tile(np.expand_dims(repeated_states, 0), [hidden_states.shape[0], 1, 1, 1])).to(torch.float)
    predictor_input = torch.concat([repeated_hidden, repeated_states_for_hidden], dim=-1).to(device)
    return predictor_input, repeated_labels, (x_, y_, z_)

def generate_predictor_wind_input(hidden_states, tasks, reward_radius=0.05, train=False):
    l = 50
    z_ = 0#0.15 / 2
    n = 10
    if train:
        r = 0.15
        theta = np.linspace(-np.pi, np.pi, l)
        x_ = np.cos(theta) * r
        y_ = np.sin(theta) * r
        all_states = (torch.normal(0, 0.05 / 3, (n, 1, 3)).numpy() + tasks.unsqueeze(0).cpu().numpy()).reshape(-1, 3)
        r = 0.3
        x_ = y_ = np.linspace(-r, r, l)
        x, y, z = np.meshgrid(x_, y_, z_)
        square_states = np.dstack((x, y, z)).reshape(-1, 3)
        idx = np.arange(len(square_states))
        np.random.shuffle(idx)
        all_states = np.vstack([all_states, square_states[idx[:(n * l) // 5]]])

    else:
        r = 0.3
        x_ = y_ = np.linspace(-r, r, l)
        x, y, z = np.meshgrid(x_, y_, z_)
        all_states = np.dstack((x, y, z)).reshape(-1, 3)

    repeated_tasks = tasks.unsqueeze(1).repeat(1, all_states.shape[0], 1).detach().cpu().numpy()
    repeated_states = np.tile(np.expand_dims(all_states, 0), [tasks.shape[0], 1, 1])
    labels = torch.tensor(np.linalg.norm(repeated_tasks - repeated_states, axis=-1) <= reward_radius).to(torch.float)
    repeated_labels = labels.unsqueeze(0).repeat(hidden_states.shape[0], 1, 1).unsqueeze(-1).to(device)
    repeated_hidden = torch.tensor(hidden_states.unsqueeze(-2).repeat(1, 1, all_states.shape[0], 1)).cpu()
    repeated_states_for_hidden = torch.tensor(
        np.tile(np.expand_dims(repeated_states, 0), [hidden_states.shape[0], 1, 1, 1])).to(torch.float)
    predictor_input = torch.concat([repeated_hidden, repeated_states_for_hidden], dim=-1).to(device)
    return predictor_input, repeated_labels, (x_, y_, z_)


def collect_data(env, args, policy, encoder=None ):
    num_episodes = args.max_rollouts_per_task

    # --- initialise things we want to keep track of ---

    episode_prev_obs = [[] for _ in range(num_episodes)]
    episode_next_obs = [[] for _ in range(num_episodes)]
    episode_actions = [[] for _ in range(num_episodes)]
    episode_rewards = [[] for _ in range(num_episodes)]

    episode_returns = []
    episode_lengths = []

    if encoder is not None:
        episode_hidden_states = [[] for _ in range(num_episodes)]
    else:
        episode_hidden_states = None

    # --- roll out policy ---

    # (re)set environment
    state, task = reset_env(env, args)
    start_obs_raw = state.clone()
    # task = task.view(-1) if task is not None else None

    # initialise actions and rewards (used as initial input to policy if we have a recurrent policy)
    if hasattr(args, 'hidden_size'):
        hidden_state = torch.zeros((1, args.hidden_size)).to(device)
    else:
        hidden_state = None
    pos = [[] for _ in range(args.max_rollouts_per_task)]
    start_pos = env.get_pos()

    # keep track of what task we're in and the position of the cheetah
    for episode_idx in range(num_episodes):

        curr_rollout_rew = []
        pos[episode_idx].append(start_pos)
        if episode_idx == 0:
            if encoder is not None:
                # reset to prior
                current_hidden_state = encoder.prior(args.num_processes)
                current_hidden_state.to(device)
        episode_hidden_states[episode_idx].append(current_hidden_state[0].clone())

        for step_idx in range(1, env._max_episode_steps + 1):

            if step_idx == 1:
                episode_prev_obs[episode_idx].append(start_obs_raw.clone())
            else:
                episode_prev_obs[episode_idx].append(state.clone())
            # act
            _, action = select_action_cpc(args=args, policy=policy, deterministic=True,
                                          hidden_latent=current_hidden_state.squeeze(0), state=state, task=task)

            (state, task), (rew, rew_normalised), done, info = env_step(env, action, args)
            state = state.float().to(device)
            pos[episode_idx].append(env.get_pos())
            if encoder is not None:
                # update task embedding
                current_hidden_state = encoder(
                    action.reshape(args.num_processes, -1).float().to(device), state,
                    rew.reshape(args.num_processes, -1).float().to(device),
                    current_hidden_state, return_prior=False)

                episode_hidden_states[episode_idx].append(current_hidden_state[0].clone())

            episode_next_obs[episode_idx].append(state.clone())
            episode_rewards[episode_idx].append(rew.clone())
            episode_actions[episode_idx].append(action.clone())
            if info[0]['done_mdp'] and not done[0]:
                start_obs_raw = np.stack([inf['start_state'] for inf in info])
                start_obs_raw = torch.from_numpy(start_obs_raw).float().to(device)
                start_pos = env.get_pos()
                break

        episode_returns.append(sum(curr_rollout_rew))
        episode_lengths.append(step_idx)

    # clean up

    episode_pos = np.concatenate([np.stack(p, axis=1)[:, :50, ...] for p in pos], axis=1)
    episode_prev_obs = torch.concat([torch.stack(e ,dim=1) for e in episode_prev_obs],dim=1)
    episode_next_obs = torch.concat([torch.stack(e, dim=1) for e in episode_next_obs], dim=1)
    episode_hidden_states = torch.concat([torch.stack(e, dim=1)[:,:50,...] for e in episode_hidden_states],dim=1)
    episode_actions = torch.concat([torch.stack(e, dim=1) for e in episode_actions], dim=1)
    episode_rewards = torch.concat([torch.cat(e, dim=1) for e in episode_rewards], dim=1)
    return episode_hidden_states, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
           episode_returns, episode_pos, task

def sample_goal_func(pos, negative_sampling_factor):
    a = np.random.random(size=(*pos.shape[:-1], negative_sampling_factor)) * 2 * np.pi
    r = 3 * np.random.random(size=(*pos.shape[:-1], negative_sampling_factor)) ** 0.5
    return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

def relabel_func(pos, negative_sampling_factor, rewards, tasks):
    goals = sample_goal_func(pos, negative_sampling_factor)
    pos = pos.unsqueeze(-2).tile(1,1,negative_sampling_factor,1)
    tasks = tasks.unsqueeze(0).unsqueeze(2).tile(pos.shape[0], 1, pos.shape[2], 1)
    rewards = rewards.unsqueeze(-2).tile(1,1, negative_sampling_factor, 1)
    rewards += torch.abs(pos - tasks).sum(dim=-1, keepdim=True)
    rewards -= torch.abs(pos - torch.from_numpy(goals).to(device=device)).sum(dim=-1, keepdim=True)
    return rewards


# ±def get_linear_layer_memory_bytes(action_size, embedding_size):
#     return 4*action_size * embedding_size
#
# def get_conv_layer_memory_bytes(d, hidden_sizes):
#     total_params = 0
#     from math import floor
#     x = d
#     dilation = 1
#     padding = 0
#     input_channels = 3
#     images_size = (x // 2)**2 * hidden_sizes[0][0] + (x // 4)**2 *hidden_sizes[1][0]
#     for hidden in hidden_sizes:
#         out_channels, kernel, stride = hidden
#         params_per_filter = input_channels * (kernel**2) + 1
#         total_params += out_channels * params_per_filter
#         x = floor(((x + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1)
#         input_channels = out_channels
#     return 4*images_size, 4*params_per_filter
#
# def get_memory_consumption_mb(d, trajectory_len):
#     image_size, param_size = get_conv_layer_memory_bytes(d, [(32,3,2), (32,3,2), (32,3,1)])
#     action_size = get_linear_layer_memory_bytes(2, 16)
#     total_bytes = image_size*trajectory_len + 3*param_size + action_size
#     return total_bytes // (1024**2)
#
#
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from scipy import stats
# import seaborn as sns
# sns.color_palette("dark")
# import matplotlib.pyplot as plt
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
# set_default_plot_params()
# memory_consumption = []
# im_sizes = np.arange(64, 150, 20)
# for im_size in im_sizes:
#     memory_consumption.append(get_memory_consumption_mb(im_size, 200))
# plt.plot(im_sizes, memory_consumption, marker='o')
# plt.xlabel('Image dimension')
# plt.ylabel('Memory consumption (MiB)')
# plt.title('VariBAD Memory Consumption')
# plt.grid()
# plt.savefig('VariBAD_mem')