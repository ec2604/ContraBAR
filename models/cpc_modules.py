import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, latent_dim, lookahead_factor=1):
        super(MLP, self).__init__()
        # Check input size
        self.ln = nn.LayerNorm(latent_dim)
        self.hidden1 = nn.Linear(latent_dim, latent_dim // 2)
        self.activation = nn.ELU()
        self.hidden2 = nn.Linear(latent_dim // 2, lookahead_factor)

    def forward(self, x):
        out = self.ln(x)
        out = self.activation(self.hidden1(out))
        out = self.hidden2(out)
        return out


class CPCMatrix(nn.Module):
    def __init__(self, context_dim, z_dim):
        super(CPCMatrix, self).__init__()
        self.mat = nn.Linear(context_dim, z_dim)

    def forward(self, context, z):
        return torch.matmul(self.mat(context).unsqueeze(-2), z.unsqueeze(-1))


class actionGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(actionGRU, self).__init__()
        # Check input size
        self.ln = torch.nn.LayerNorm(input_dim)
        self.gru1 = nn.GRU(input_dim, hidden_dim)

    def forward(self, x, hidden):
        out = self.ln(x)
        out, _ = self.gru1(out, hidden)
        return out, _


class statePredictor(nn.Module):
    def __init__(self, input_dim, state_dim):
        super(statePredictor, self).__init__()
        # Check input size
        self.hidden1 = nn.Linear(input_dim, 1024)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.hidden3 = nn.Linear(1024, state_dim)

    def forward(self, x):
        out = self.relu1(self.hidden1(x))
        out = self.relu2(self.hidden2(out))
        out = self.hidden3(out)
        return out
