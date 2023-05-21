

## Very untidy scripts for using an MLP classifier to prove belief.

import torch
import numpy as np
from models.cpc_modules import statePredictor
from utils.helpers import generate_predictor_input as gpi
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## for panda reacher
def generate_panda_train_input(hidden_states, tasks):
    r = 0.3
    z_ = 0.15 / 2
    l = 100
    x_ = y_ = np.linspace(-r, r, l)
    x, y, z = np.meshgrid(x_, y_, z_)
    square_states = np.dstack((x, y, z)).reshape(-1, 3).astype(np.float32)
    np.random.shuffle(square_states)
    state_idx = np.random.choice(np.arange(square_states.shape[0]), replace=True,
                                 size=hidden_states.shape[0] * hidden_states.shape[1])
    repeated_tasks = tasks.unsqueeze(1).repeat(1, hidden_states.shape[1], 1)
    repeated_tasks = repeated_tasks + torch.normal(0, 0.05/4,size=repeated_tasks.shape).to(device)
    square_states = torch.from_numpy(square_states[state_idx].reshape(*repeated_tasks.shape)).to(device)
    positives = torch.cat([hidden_states, repeated_tasks], dim=-1)
    negatives = torch.cat([hidden_states, square_states], dim=-1)
    return positives, negatives

def generate_panda_wind_train_input(hidden_states, tasks):
    r = 0.5
    z_ = 0#0.15 / 2
    l = 50
    x_ = y_ = np.linspace(-r, r, l)
    x, y, z = np.meshgrid(x_, y_, z_)
    square_states = np.dstack((x, y, z)).reshape(-1, 3).astype(np.float32)
    np.random.shuffle(square_states)
    state_idx = np.random.choice(np.arange(square_states.shape[0]), replace=True,
                                 size=hidden_states.shape[0] * hidden_states.shape[1])
    repeated_tasks = tasks.unsqueeze(1).repeat(1, hidden_states.shape[1], 1)
    square_states = torch.from_numpy(square_states[state_idx].reshape(*repeated_tasks.shape)).to(device)
    positives = torch.cat([hidden_states, repeated_tasks], dim=-1)
    negatives = torch.cat([hidden_states, square_states], dim=-1)
    return positives, negatives

def generate_predictor_input(hidden_states, tasks, reward_radius=0.05, train=False):
    l = 100
    r = 0.3
    z_ = 0.15 / 2
    x_ = y_ = np.linspace(-r, r, l)
    x, y, z = np.meshgrid(x_, y_, z_)
    all_states = np.dstack((x, y, z)).reshape(-1, 3)

    repeated_tasks = tasks.unsqueeze(1).repeat(1, all_states.shape[0], 1).detach().cpu().numpy()
    repeated_states = np.tile(np.expand_dims(all_states, 0), [tasks.shape[0], 1, 1])
    labels = torch.tensor(np.linalg.norm(repeated_tasks - repeated_states, axis=-1) <= reward_radius).to(torch.float)
    repeated_labels = labels.unsqueeze(1).repeat(1,hidden_states.shape[1], 1).unsqueeze(-1).to(device)
    repeated_hidden = hidden_states.unsqueeze(-2).repeat(1, 1, all_states.shape[0], 1).detach().cpu()
    repeated_states_for_hidden = torch.tensor(
        np.tile(np.expand_dims(repeated_states, 1), [1,hidden_states.shape[1], 1, 1])).to(torch.float)
    predictor_input = torch.concat([repeated_hidden, repeated_states_for_hidden], dim=-1).to(device)
    return predictor_input, repeated_labels, (x_, y_, z_)


if __name__ == '__main__':
    hidden_size = 50
    task_size = 3
    lr = 8e-3
    representation_evaluator = statePredictor(hidden_size + task_size, 1).to(device)
    representation_evaluation_loss = torch.nn.BCEWithLogitsLoss()
    representation_evaluator_optimizer = torch.optim.Adam(representation_evaluator.parameters(),
                                                          lr=lr)
    data = np.load('/mnt/data/erac/logs_CustomReachWind-v0/panda_wind_belief_ds_seed_88.npz')#np.load('/mnt/data/erac/logs_CustomReach-v0/panda_wind_belief_ds_seed_seed_16.npz')
    hidden = torch.from_numpy(data['hidden_buffer'][:-16])
    task = torch.from_numpy(data['task_buffer'])[:-16][...,3:]
    traj_idx = np.arange(len(hidden))
    np.random.shuffle(traj_idx)

    train_split = int(0.8 * (len(traj_idx)))
    train_hidden = hidden[traj_idx[:train_split]]
    train_task = task[traj_idx[:train_split]]

    test_hidden = hidden[traj_idx[train_split:]]
    test_task = task[traj_idx[train_split:]]
    num_batches = 5000
    # num_batches = 2700
    batch_size = 512
    for i in range(num_batches):
        batch_idx = np.random.choice(np.arange(train_split), size=batch_size, replace=False)
        batch_hidden = train_hidden[batch_idx].to(device)
        batch_task = train_task[batch_idx].to(device)
        positives, negatives = generate_panda_wind_train_input(batch_hidden, batch_task)
        positive_loss = representation_evaluation_loss(representation_evaluator(positives).reshape(-1,1),
                                                         torch.ones(positives.shape[:2],device=device).float().reshape(-1,1))
        negative_loss = representation_evaluation_loss(representation_evaluator(negatives).reshape(-1,1),
                                                         torch.zeros(negatives.shape[:2],device=device).float().reshape(-1, 1))
        loss = positive_loss + negative_loss
        representation_evaluator_optimizer.zero_grad()
        loss.backward()
        representation_evaluator_optimizer.step()
        if (i % 10) == 0:
            print(f'Positive loss, i={i}: {positive_loss}',flush=True)
            print(f'Train loss, i={i}: {loss}',flush=True)
        if (i % 50) == 0:
            batch_hidden = test_hidden.to(device)
            batch_task = test_task.to(device)
            positives, negatives = generate_panda_wind_train_input(batch_hidden, batch_task)
            positive_loss = representation_evaluation_loss(representation_evaluator(positives).reshape(-1, 1),
                                                           torch.ones(positives.shape[:2],
                                                                      device=device).float().reshape(-1, 1))
            negative_loss = representation_evaluation_loss(representation_evaluator(negatives).reshape(-1, 1),
                                                           torch.zeros(negatives.shape[:2],
                                                                       device=device).float().reshape(-1, 1))
            eval_loss = positive_loss + negative_loss
            print(f'===Eval loss, i={i}: {eval_loss}',flush=True)
            print(f'===Diff, i={i}: {loss - eval_loss}',flush=True)
        # torch.save(representation_evaluator, "/mnt/data/erac/logs_CustomReach-v0/belief_panda_seed_16.pt")
        torch.save(representation_evaluator, "/mnt/data/erac/logs_CustomReachWind-v0/belief_panda_seed_88.pt")

