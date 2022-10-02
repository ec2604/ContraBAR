import matplotlib.pyplot as plt
import numpy as np
import torch

from environments.parallel_envs import make_vec_envs
from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def evaluate(args,
#              policy,
#              ret_rms,
#              iter_idx,
#              tasks,
#              encoder=None,
#              num_episodes=None,
#              **kwargs
#              ):
#     env_name = args.env_name
#     if hasattr(args, 'test_env_name'):
#         env_name = args.test_env_name
#     if num_episodes is None:
#         num_episodes = args.max_rollouts_per_task
#     num_processes = args.num_processes
#
#     # --- set up the things we want to log ---
#
#     # for each process, we log the returns during the first, second, ... episode
#     # (such that we have a minimum of [num_episodes]; the last column is for
#     #  any overflow and will be discarded at the end, because we need to wait until
#     #  all processes have at least [num_episodes] many episodes)
#     returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)
#
#     episode_returns = []
#     episode_lengths = []
#
#     # --- initialise environments and latents ---
#
#     envs = make_vec_envs(env_name,
#                          seed=args.seed * 42 + iter_idx,
#                          num_processes=num_processes,
#                          gamma=args.policy_gamma,
#                          device=device,
#                          rank_offset=num_processes + 1,  # to use diff tmp folders than main processes
#                          episodes_per_task=num_episodes,
#                          normalise_rew=args.norm_rew_for_policy,
#                          ret_rms=ret_rms,
#                          tasks=tasks,
#                          add_done_info=args.max_rollouts_per_task > 1,
#                          dummy=True
#                          )
#     num_steps = envs._max_episode_steps
#     prev_state_per_episode = torch.zeros((num_episodes, num_steps, num_processes, *args.state_dim)).to(device)
#     next_state_per_episode = torch.zeros((num_episodes, num_steps, num_processes, *args.state_dim)).to(device)
#     actions_per_episode = torch.zeros((num_episodes, num_steps, num_processes, args.action_dim)).to(device)
#     rewards_per_episode = torch.zeros((num_episodes, num_steps, num_processes, 1)).to(device)
#     tasks = torch.zeros((num_processes, args.task_dim)).to(device)
#     # reset environments
#     state, task = utl.reset_env(envs, args)
#     tasks = task.clone()
#     # this counts how often an agent has done the same task already
#     task_count = torch.zeros(num_processes).long().to(device)
#
#     if encoder is not None:
#         # reset latent state to prior
#         hidden_state = encoder.encoder.prior(num_processes)
#     else:
#         hidden_state = None
#
#     for episode_idx in range(num_episodes):
#
#         for step_idx in range(num_steps):
#             prev_state_per_episode[episode_idx, step_idx, :, ...] = state.clone()
#             with torch.no_grad():
#                 _, action = utl.select_action_cpc(args=args, policy=policy, deterministic=True,
#                                                   hidden_latent=hidden_state.squeeze(0), state=state, task=task)
#
#             # observe reward and next obs
#             [state, task], (rew_raw, rew_normalised), done, infos = utl.env_step(envs, action, args)
#             done_mdp = [info['done_mdp'] for info in infos]
#
#             if encoder is not None:
#                 # update the hidden state
#                 # hidden_state = utl.update_encoding_cpc(encoder=encoder.encoder,
#                 #                                        next_obs=state,
#                 #                                        action=action,
#                 #                                        reward=rew_raw,
#                 #                                        done=None,
#                 #                                        hidden_state=hidden_state)
#                 hidden_state = encoder.encoder(
#                     action.float().to(device), state, rew_raw.float().to(device),
#                     hidden_state, return_prior=False)
#
#             next_state_per_episode[episode_idx, step_idx, :, ...] = state.clone()
#             rewards_per_episode[episode_idx, step_idx, :, ...] = rew_raw.clone()
#             actions_per_episode[episode_idx, step_idx, :, ...] = action.clone()
#             # add rewards
#             returns_per_episode[range(num_processes), task_count] += rew_raw.view(-1)
#
#             for i in np.argwhere(done_mdp).flatten():
#                 # count task up, but cap at num_episodes + 1
#                 task_count[i] = min(task_count[i] + 1, num_episodes)  # zero-indexed, so no +1
#             if np.sum(done) > 0:
#                 done_indices = np.argwhere(done.flatten()).flatten()
#                 state, task = utl.reset_env(envs, args, indices=done_indices, state=state)
#
#     envs.close()
#     prev_state_per_episode = prev_state_per_episode.reshape(num_episodes * num_steps, num_processes, *args.state_dim)
#     next_state_per_episode = next_state_per_episode.reshape(num_episodes * num_steps, num_processes, *args.state_dim)
#     actions_per_episode = actions_per_episode.reshape(num_episodes * num_steps, num_processes, args.action_dim)
#     rewards_per_episode = rewards_per_episode.reshape(num_episodes * num_steps, num_processes, 1)
#
#     eval_cpc_stats = encoder.compute_cpc_loss(batch=
#     (prev_state_per_episode, next_state_per_episode, actions_per_episode, rewards_per_episode, tasks, \
#      num_steps * num_episodes))
#     eval_representation_stats = None
#     if args.evaluate_representation:
#         eval_representation_stats = encoder.calc_latent_evaluator_loss(tasks, hidden_state) # fix this
#     return returns_per_episode[:, :num_episodes], eval_cpc_stats, eval_representation_stats
def evaluate(args,
             policy,
             ret_rms,
             iter_idx,
             tasks,
             encoder=None,
             num_episodes=None,
             **kwargs
             ):
    env_name = args.env_name
    if hasattr(args, 'test_env_name'):
        env_name = args.test_env_name
    if num_episodes is None:
        num_episodes = args.max_rollouts_per_task
    num_processes = args.num_processes

    # --- set up the things we want to log ---

    # for each process, we log the returns during the first, second, ... episode
    # (such that we have a minimum of [num_episodes]; the last column is for
    #  any overflow and will be discarded at the end, because we need to wait until
    #  all processes have at least [num_episodes] many episodes)
    returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)

    episode_returns = []
    episode_lengths = []

    # --- initialise environments and latents ---
    prev_state_per_episode_list = []
    next_state_per_episode_list = []
    actions_per_episode_list = []
    rewards_per_episode_list = []
    if len(args.underlying_state_dim) > 0:
        underlying_state_per_episode_list = []
    tasks_list = []
    hidden_state_list_all_proc = []
    for i in range(num_processes):
        envs = make_vec_envs(env_name,
                             seed=args.seed * 42 + iter_idx,
                             num_processes=1,
                             gamma=args.policy_gamma,
                             device=device,
                             rank_offset=num_processes + 1,  # to use diff tmp folders than main processes
                             episodes_per_task=num_episodes,
                             normalise_rew=args.norm_rew_for_policy,
                             ret_rms=ret_rms,
                             tasks=tasks,
                             add_done_info=args.max_rollouts_per_task > 1,
                             dummy=True
                             )
        num_steps = envs._max_episode_steps
        prev_state_per_episode = torch.zeros((num_episodes, num_steps, *args.state_dim)).to(device)
        # prev_state_per_episode = torch.zeros((num_episodes, num_steps, 4, 64, 64)).to(device)
        next_state_per_episode = torch.zeros((num_episodes, num_steps, *args.state_dim)).to(device)
        # next_state_per_episode = torch.zeros((num_episodes, num_steps, 4, 64, 64)).to(device)

        actions_per_episode = torch.zeros((num_episodes, num_steps, args.action_dim)).to(device)
        rewards_per_episode = torch.zeros((num_episodes, num_steps, 1)).to(device)
        if len(args.underlying_state_dim) >  0:
            underlying_state_per_episode = torch.zeros((num_episodes, num_steps, *args.underlying_state_dim)).to(device)
        hidden_state_list = []
        # reset environments
        state, task = utl.reset_env(envs, args)
        # orig_state  = torch.from_numpy(
        #             np.stack([method for method in envs.venv.get_images_()])).to(device)
        # orig_state = torch.cat([orig_state, torch.ones(orig_state.shape[0], 1,
        #                                                              *orig_state.shape[2:]).to(
        #     device)], dim=1).float()
        tasks_list.append(task.clone())
        # this counts how often an agent has done the same task already
        task_count = 0

        if encoder is not None:
            # reset latent state to prior
            hidden_state = encoder.encoder.prior(1)
        else:
            hidden_state = None
        hidden_state_list.append(hidden_state.clone())
        for episode_idx in range(num_episodes):
            curr_returns_per_episode = torch.zeros((1)).to(device)
            for step_idx in range(num_steps):
                prev_state_per_episode[episode_idx, step_idx, :, ...] = state.clone()
                with torch.no_grad():
                    _, action = utl.select_action_cpc(args=args, policy=policy, deterministic=True,
                                                      hidden_latent=hidden_state.squeeze(0), state=state, task=task)

                # observe reward and next obs
                [state, task], (rew_raw, rew_normalised), done, infos = utl.env_step(envs, action, args)
                done_mdp = [info['done_mdp'] for info in infos]
                # if len(args.underlying_state_dim) > 0:
                    # underlying_states = torch.from_numpy(np.stack([info['state'] for info in infos],axis=0)).to(device)
                # underlying_states = torch.from_numpy(
                #     np.stack([method for method in envs.venv.get_images_()])).to(device)
                # if done[0]:
                #     underlying_states = torch.cat([underlying_states, torch.ones(underlying_states.shape[0], 1,
                #                                                                  *underlying_states.shape[2:]).to(
                #         device)], dim=1).float()
                # else:
                #     underlying_states = torch.cat([underlying_states, torch.zeros(underlying_states.shape[0], 1,
                #                                                                   *underlying_states.shape[2:]).to(
                #         device)],
                #                                   dim=1).float()
                # rew_cpc = torch.from_numpy(np.stack([info['goal_forward'] >= 0.3 for info in infos],axis=0)).to(device)
                if encoder is not None:
                    # update the hidden state
                    # hidden_state = utl.update_encoding_cpc(encoder=encoder.encoder,
                    #                                        next_obs=state,
                    #                                        action=action,
                    #                                        reward=rew_raw,
                    #                                        done=None,
                    #                                        hidden_state=hidden_state)
                    hidden_state = encoder.encoder(
                        action.float().to(device), state,
                        # rew_cpc.reshape(-1,1).float().to(device),
                        rew_raw.float().to(device),
                        hidden_state, return_prior=False)
                    hidden_state_list.append(hidden_state.clone())

                next_state_per_episode[episode_idx, step_idx, ...] = state.clone()
                rewards_per_episode[episode_idx, step_idx, ...] = rew_raw.clone()
                actions_per_episode[episode_idx, step_idx, ...] = action.clone()
                # if len(args.underlying_state_dim) > 0:
                #     underlying_state_per_episode[episode_idx, step_idx, ...] = underlying_states.clone()
                # add rewards
                curr_returns_per_episode += rew_raw.view(-1).clone()
                returns_per_episode[i, task_count] = curr_returns_per_episode.clone()
                for j in np.argwhere(done_mdp).flatten():
                    # count task up, but cap at num_episodes + 1
                    task_count = min(task_count + 1, num_episodes)  # zero-indexed, so no +1
                if np.sum(done) > 0:
                    done_indices = np.argwhere(done.flatten()).flatten()
                    state, task = utl.reset_env(envs, args, indices=done_indices, state=state)


        envs.close()
        # if len(args.underlying_state_dim) > 0:
        #     underlying_state_per_episode_list.append(underlying_state_per_episode.clone())
        rewards_per_episode_list.append(rewards_per_episode.clone())
        prev_state_per_episode_list.append(prev_state_per_episode.clone())
        next_state_per_episode_list.append(next_state_per_episode.clone())
        actions_per_episode_list.append(actions_per_episode.clone())
        hidden_state_list_all_proc.append(torch.cat(hidden_state_list, dim=1).clone())

    rewards_per_episode_list = torch.stack(rewards_per_episode_list, dim=2)
    if len(args.underlying_state_dim) > 0:
        underlying_state_per_episode_list = torch.stack(underlying_state_per_episode_list, dim=2)
    actions_per_episode_list = torch.stack(actions_per_episode_list, dim=2)
    prev_state_per_episode_list = torch.stack(prev_state_per_episode_list, dim=2)
    next_state_per_episode_list = torch.stack(next_state_per_episode_list, dim=2)
    hidden_state_list_all_proc = torch.cat(hidden_state_list_all_proc, dim=0)
    hidden_state_list_all_proc = torch.permute(hidden_state_list_all_proc, [1, 0, 2])
    tasks_list = torch.cat(tasks_list)

    prev_state_per_episode_list = prev_state_per_episode_list.reshape(num_episodes * num_steps, num_processes, *args.state_dim)
    next_state_per_episode_list = next_state_per_episode_list.reshape(num_episodes * num_steps, num_processes, *args.state_dim)
    actions_per_episode_list = actions_per_episode_list.reshape(num_episodes * num_steps, num_processes,
                                                                args.action_dim)
    rewards_per_episode_list = rewards_per_episode_list.reshape(num_episodes * num_steps, num_processes, 1)
    if len(args.underlying_state_dim) > 0:
        underlying_state_per_episode_list = underlying_state_per_episode_list.reshape(num_episodes * num_steps, num_processes,
                                                                                      *args.underlying_state_dim)
    else:
        underlying_state_per_episode_list = None
    eval_cpc_stats = encoder.compute_cpc_loss(batch=
                                              (prev_state_per_episode_list, next_state_per_episode_list, actions_per_episode_list,
                                               rewards_per_episode_list, tasks_list, underlying_state_per_episode_list,\
                                               num_steps * num_episodes))
    eval_representation_stats = None
    if args.evaluate_representation and args.evaluate_start_iter <= iter_idx:
        eval_representation_stats = encoder.calc_latent_evaluator_loss(tasks_list, hidden_state_list_all_proc)  # fix this
    return returns_per_episode[:, :num_episodes], eval_cpc_stats, eval_representation_stats


def visualise_behaviour(args,
                        policy,
                        image_folder,
                        iter_idx,
                        ret_rms,
                        tasks,
                        encoder=None,
                        reward_decoder=None,
                        logger=None
                        ):
    # initialise environment
    env = make_vec_envs(env_name=args.env_name,
                        seed=args.seed * 42 + iter_idx,
                        num_processes=1,
                        gamma=args.policy_gamma,
                        device=device,
                        episodes_per_task=args.max_rollouts_per_task,
                        normalise_rew=args.norm_rew_for_policy, ret_rms=ret_rms,
                        rank_offset=args.num_processes + 42,  # not sure if the temp folders would otherwise clash
                        tasks=tasks
                        )
    episode_task = torch.from_numpy(np.array(env.get_task())).to(device).float()

    # get a sample rollout
    unwrapped_env = env.venv.unwrapped.envs[0]
    if hasattr(env.venv.unwrapped.envs[0], 'unwrapped'):
        unwrapped_env = unwrapped_env.unwrapped
    if hasattr(unwrapped_env, 'visualise_behaviour'):
        # if possible, get it from the env directly
        # (this might visualise other things in addition)
        traj = unwrapped_env.visualise_behaviour(env=env,
                                                 args=args,
                                                 policy=policy,
                                                 iter_idx=iter_idx,
                                                 encoder=encoder.encoder,
                                                 image_folder=image_folder,
                                                 belief_evaluator=encoder.representation_evaluator if args.evaluate_representation else None,
                                                 logger=logger,
                                                 )
    else:
        traj = get_test_rollout(args, env, policy, encoder)

    hidden_states, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, episode_returns = traj

    # if hidden_states is not None:
    # plot_latents(latent_means, latent_logvars,
    #              image_folder=image_folder,
    #              iter_idx=iter_idx
    #              )

    # if not (args.disable_decoder and args.disable_kl_term):
    #     pass
    # plot_vae_loss(args,
    #               hidden_states,
    #               episode_prev_obs,
    #               episode_next_obs,
    #               episode_actions,
    #               episode_rewards,
    #               episode_task,
    #               image_folder=image_folder,
    #               iter_idx=iter_idx
    #               )

    env.close()


def get_test_rollout(args, env, policy, encoder=None):
    num_episodes = args.max_rollouts_per_task

    # --- initialise things we want to keep track of ---

    episode_prev_obs = [[] for _ in range(num_episodes)]
    episode_next_obs = [[] for _ in range(num_episodes)]
    episode_actions = [[] for _ in range(num_episodes)]
    episode_rewards = [[] for _ in range(num_episodes)]

    episode_returns = []
    episode_lengths = []

    episode_hidden_states = [[] for _ in range(num_episodes)]

    # --- roll out policy ---

    # (re)set environment
    env.reset_task()
    state, task = utl.reset_env(env, args)
    state = state.to(device)
    task = task.view(-1) if task is not None else None

    for episode_idx in range(num_episodes):

        curr_rollout_rew = []

        if encoder is not None:
            if episode_idx == 0:
                # reset to prior
                curr_hidden_state = encoder.encoder.prior(1)

            episode_hidden_states[episode_idx].append(curr_hidden_state[0].clone())
        for step_idx in range(1, env._max_episode_steps + 1):

            episode_prev_obs[episode_idx].append(state.clone())

            _, action = policy.act(state=state, latent=curr_hidden_state, task=task, deterministic=True)
            action = action.reshape((1, *action.shape))

            # observe reward and next obs
            (state, task), (rew_raw, rew_normalised), done, infos = utl.env_step(env, action, args)
            state = state.reshape((1, -1)).to(device)
            #task = task.view(-1) if task is not None else None

            if encoder is not None:
                # update task embedding
                curr_hidden_state = encoder(
                    action.float().to(device),
                    state,
                    rew_raw.reshape((1, 1)).float().to(device),
                    curr_hidden_state,
                    return_prior=False)

                episode_hidden_states[episode_idx].append(curr_hidden_state[0].clone())

            episode_next_obs[episode_idx].append(state.clone())
            episode_rewards[episode_idx].append(rew_raw.clone())
            episode_actions[episode_idx].append(action.clone())

            if infos[0]['done_mdp']:
                break

        episode_returns.append(sum(curr_rollout_rew))
        episode_lengths.append(step_idx)

    # clean up
    if encoder is not None:
        episode_hidden_states = [torch.stack(e) for e in episode_hidden_states]

    episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
    episode_next_obs = [torch.cat(e) for e in episode_next_obs]
    episode_actions = [torch.cat(e) for e in episode_actions]
    episode_rewards = [torch.cat(r) for r in episode_rewards]

    return episode_hidden_states, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
           episode_returns
