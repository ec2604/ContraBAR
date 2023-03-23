import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from math import floor
from utils import helpers as utl
from torch.nn.utils import spectral_norm
from models import GRU_layernorm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNNEncoder(nn.Module):
    def __init__(self,
                 args,
                 # network size
                 layers_before_gru=(),
                 hidden_size=64,
                 layers_after_gru=(),
                 latent_dim=32,
                 # actions, states, rewards
                 action_dim=2,
                 action_embed_dim=10,
                 state_dim=2,
                 state_embed_dim=10,
                 reward_size=1,
                 reward_embed_size=5,
                 ):
        super(RNNEncoder, self).__init__()

        self.args = args
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.reparameterise = self._sample_gaussian

        # embed action, state, reward
        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, torch.relu)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, torch.relu)
        self.reward_encoder = utl.FeatureExtractor(reward_size, reward_embed_size, torch.relu)

        # fully connected layers before the recurrent cell
        curr_input_dim = action_embed_dim + state_embed_dim + reward_embed_size
        self.fc_before_gru = nn.ModuleList([])
        for i in range(len(layers_before_gru)):
            self.fc_before_gru.append(nn.Linear(curr_input_dim, layers_before_gru[i]))
            curr_input_dim = layers_before_gru[i]

        # recurrent unit
        # TODO: TEST RNN vs GRU vs LSTM
        self.gru = nn.GRU(input_size=curr_input_dim,
                          hidden_size=hidden_size,
                          num_layers=1,
                          )

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # fully connected layers after the recurrent cell
        curr_input_dim = hidden_size
        self.fc_after_gru = nn.ModuleList([])
        for i in range(len(layers_after_gru)):
            self.fc_after_gru.append(nn.Linear(curr_input_dim, layers_after_gru[i]))
            curr_input_dim = layers_after_gru[i]

        # output layer
        self.fc_mu = nn.Linear(curr_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(curr_input_dim, latent_dim)

    def _sample_gaussian(self, mu, logvar, num=None):
        if num is None:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            raise NotImplementedError  # TODO: double check this code, maybe we should use .unsqueeze(0).expand((num, *logvar.shape))
            std = torch.exp(0.5 * logvar).repeat(num, 1)
            eps = torch.randn_like(std)
            mu = mu.repeat(num, 1)
            return eps.mul(std).add_(mu)

    def reset_hidden(self, hidden_state, done):
        """ Reset the hidden state where the BAMDP was done (i.e., we get a new task) """
        if hidden_state.dim() != done.dim():
            if done.dim() == 2:
                done = done.unsqueeze(0)
            elif done.dim() == 1:
                done = done.unsqueeze(0).unsqueeze(2)
        hidden_state = hidden_state * (1 - done)
        return hidden_state

    def prior(self, batch_size, sample=True):

        # TODO: add option to incorporate the initial state

        # we start out with a hidden state of zero
        hidden_state = torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(device)

        h = hidden_state
        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_gru)):
            h = F.relu(self.fc_after_gru[i](h))

        # outputs
        latent_mean = self.fc_mu(h)
        latent_logvar = self.fc_logvar(h)
        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        return latent_sample, latent_mean, latent_logvar, hidden_state

    def forward(self, actions, states, rewards, hidden_state, return_prior, sample=True, detach_every=None):
        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        For one-step predictions, sequence_len=1 and hidden_state!=None.
        For feeding in entire trajectories, sequence_len>1 and hidden_state=None.
        In the latter case, we return embeddings of length sequence_len+1 since they include the prior.
        """

        # we do the action-normalisation (the the env bounds) here
        actions = utl.squash_action(actions, self.args)

        # shape should be: sequence_len x batch_size x hidden_size
        actions = actions.reshape((-1, *actions.shape[-2:]))
        states = states.reshape((-1, *states.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))
        if hidden_state is not None:
            # if the sequence_len is one, this will add a dimension at dim 0 (otherwise will be the same)
            hidden_state = hidden_state.reshape((-1, *hidden_state.shape[-2:]))

        if return_prior:
            # if hidden state is none, start with the prior
            prior_sample, prior_mean, prior_logvar, prior_hidden_state = self.prior(actions.shape[1])
            hidden_state = prior_hidden_state.clone()

        # extract features for states, actions, rewards
        ha = self.action_encoder(actions)
        hs = self.state_encoder(states)
        hr = self.reward_encoder(rewards)
        h = torch.cat((ha, hs, hr), dim=2)

        # forward through fully connected layers before GRU
        for i in range(len(self.fc_before_gru)):
            h = F.relu(self.fc_before_gru[i](h))

        if detach_every is None:
            # GRU cell (output is outputs for each time step, hidden_state is last output)
            output, _ = self.gru(h, hidden_state)
        else:
            output = []
            for i in range(int(np.ceil(h.shape[0] / detach_every))):
                curr_input = h[i * detach_every:i * detach_every + detach_every]  # pytorch caps if we overflow, nice
                curr_output, hidden_state = self.gru(curr_input, hidden_state)
                output.append(curr_output)
                # detach hidden state; useful for BPTT when sequences are very long
                hidden_state = hidden_state.detach()
            output = torch.cat(output, dim=0)
        gru_h = output.clone()

        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_gru)):
            gru_h = F.relu(self.fc_after_gru[i](gru_h))

        # outputs
        latent_mean = self.fc_mu(gru_h)
        latent_logvar = self.fc_logvar(gru_h)
        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        if return_prior:
            latent_sample = torch.cat((prior_sample, latent_sample))
            latent_mean = torch.cat((prior_mean, latent_mean))
            latent_logvar = torch.cat((prior_logvar, latent_logvar))
            output = torch.cat((prior_hidden_state, output))

        if latent_mean.shape[0] == 1:
            latent_sample, latent_mean, latent_logvar = latent_sample[0], latent_mean[0], latent_logvar[0]

        return latent_sample, latent_mean, latent_logvar, output


class RNNCPCEncoder(nn.Module):

    def __init__(self,
                 args,
                 # network size
                 layers_before_gru=(),
                 hidden_size=64,
                 layers_after_gru=(),
                 latent_dim=32,
                 # actions, states, rewards
                 action_dim=2,
                 action_embed_dim=10,
                 state_dim=2,
                 state_embed_dim=10,
                 reward_size=1,
                 reward_embed_size=5,
                 ):
        super(RNNCPCEncoder, self).__init__()
        self.args = args
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

        # embed action, state, reward
        if len(state_dim) > 1:
            self.state_encoder = ImageEncoder(state_dim, state_embed_dim, self.args.image_encoder_layers)
        else:
            self.state_encoder = utl.FeatureExtractor(*state_dim, state_embed_dim, F.elu)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.elu)
        self.reward_encoder = utl.FeatureExtractor(reward_size, reward_embed_size, F.elu)

        # self.init_weights()
        # fully connected layers before the recurrent cell
        curr_input_dim = action_embed_dim + state_embed_dim + reward_embed_size#self.state_encoder.output_size + reward_embed_size
        self.fc_before_gru = nn.ModuleList([])
        for i in range(len(layers_before_gru)):
            self.fc_before_gru.append(nn.Linear(curr_input_dim, layers_before_gru[i]))
            curr_input_dim = layers_before_gru[i]
        self.ln = torch.nn.LayerNorm(curr_input_dim)
        # self.ln = nn.LayerNorm(curr_input_dim)
        if self.args.with_action_gru:
            self.ln_without_action = nn.LayerNorm(curr_input_dim - action_embed_dim)
        # recurrent unit
        # TODO: TEST RNN vs GRU vs LSTM
        self.gru = nn.GRU(input_size=curr_input_dim,
                          hidden_size=hidden_size,
                          num_layers=1,
                          )
        # self.gru = GRU_layernorm.LayerNormGRUCell(input_size=curr_input_dim,
        #                   hidden_size=hidden_size)

        # # fully connected layers after the recurrent cell
        curr_input_dim = hidden_size
        self.fc_after_gru = nn.ModuleList([])
        for i in range(len(layers_after_gru)):
            self.fc_after_gru.append(nn.Linear(curr_input_dim, layers_after_gru[i]))
            curr_input_dim = layers_after_gru[i]

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu', a=math.sqrt(2))
    #             # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             # nn.init.xavier_normal_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             pass
    #             # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    #             # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=math.sqrt(5))
    #             # nn.init.xavier_uniform_(m.weight)
    #             # nn.init.xavier_normal_(m.weight)
    #             # nn.init.orthogonal_(m.weight)
    #             # nn.init.normal_(m.weight, 0, 0.01)
    #             # if m.bias is not None:
    #             #     nn.init.constant_(m.bias, 0)
    #             # m.bias.data.fill_(.0)

    def reset_hidden(self, hidden_state, done):
        """ Reset the hidden state where the BAMDP was done (i.e., we get a new task) """
        if hidden_state.dim() != done.dim():
            if done.dim() == 2:
                done = done.unsqueeze(0)
            elif done.dim() == 1:
                done = done.unsqueeze(0).unsqueeze(2)
        hidden_state = hidden_state * (1 - done)
        return hidden_state

    def prior(self, batch_size, sample=True):

        # TODO: add option to incorporate the initial state

        # we start out with a hidden state of zero
        hidden_state = torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(device)

        return hidden_state

    def embed_input(self, actions, states, rewards, with_actions=True, with_rewards=True):
        # we do the action-normalisation (the the env bounds) here
        actions = utl.squash_action(actions, self.args)
        # shape should be: sequence_len x batch_size x hidden_size
        actions = actions.reshape((-1, *actions.shape[-2:]))
        if len(states.shape) > 3:
            orig_state_shape = states.shape
            states = states.reshape((-1, *states.shape[-3:]))
            hs = self.state_encoder(states)
            hs = hs.reshape(*orig_state_shape[:-3], -1)
        else:
            states = states.reshape((-1, *states.shape[-2:]))
            hs = self.state_encoder(states)
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))

        # extract features for states, actions, rewards
        ha = self.action_encoder(actions)
        hr = self.reward_encoder(rewards)
        if len(hs.shape) == 2:
            hs = hs.unsqueeze(0)
        h = hs
        if with_rewards:
            h = torch.cat((hr, h), dim=2)
        if with_actions:
            h = torch.cat((ha, h), dim=2)
        # forward through fully connected layers before GRU
        for i in range(len(self.fc_before_gru)-1):
            h = F.relu(self.fc_before_gru[i](h))
        if len(self.fc_before_gru) > 0:
            h = self.fc_before_gru[-1](h)
        if with_actions:
            h = self.ln(h)
        else:
            h = self.ln_without_action(h)
        return h

    def forward(self, actions, states, rewards, hidden_state, return_prior, sample=True, detach_every=None):
        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        For one-step predictions, sequence_len=1 and hidden_state!=None.
        For feeding in entire trajectories, sequence_len>1 and hidden_state=None.
        In the latter case, we return embeddings of length sequence_len+1 since they include the prior.
        """
        h = self.embed_input(actions, states, rewards)
        if hidden_state is not None:
            # if the sequence_len is one, this will add a dimension at dim 0 (otherwise will be the same)
            hidden_state = hidden_state.reshape((-1, *hidden_state.shape[-2:]))

        if return_prior:
            # if hidden state is none, start with the prior
            prior_hidden_state = self.prior(actions.shape[1])
            hidden_state = prior_hidden_state.clone()

        if detach_every is None:
            # GRU cell (output is outputs for each time step, hidden_state is last output)
            output, hidden_state = self.gru(h, hidden_state)
        else:
            output_list = []
            for i in range(int(np.ceil(h.shape[0] / detach_every))):
                curr_input = h[i * detach_every:i * detach_every + detach_every]  # pytorch caps if we overflow, nice
                curr_output, hidden_state = self.gru(curr_input, hidden_state)
                # detach hidden state; useful for BPTT when sequences are very long
                hidden_state = hidden_state.detach()
                output_list.append(curr_output)
            output = torch.cat(output_list, dim=0)
        if return_prior:
            output = torch.cat((prior_hidden_state, output))
        # output = output.clone()
        return output


class ImageEncoder(nn.Module):

    def __init__(self, state_dim, state_embed_dim, hidden_sizes):
        super(ImageEncoder, self).__init__()
        self.state_dim = state_dim
        self.state_embed_dim = state_embed_dim
        self.activation_func = nn.ReLU(inplace=True)
        self.hidden_sizes = hidden_sizes  # [(channels, kernel, stride),...]
        self.convs = nn.ModuleList([])

        input_channels = state_dim[0]
        x, y = state_dim[1:]
        dilation = 1
        padding = 0

        for hidden in self.hidden_sizes:
            out_channels, kernel, stride = hidden
            conv = spectral_norm(nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=kernel,
                             stride=stride))
            self.convs.append(conv)
            input_channels = out_channels
            x = floor(((x + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1)
            y = floor(((y + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1)

        self.output_size = x * y * input_channels
        self.fc = nn.Linear(self.output_size, state_embed_dim)
        self.ln = nn.LayerNorm(state_embed_dim)

    def forward(self, input):
        out = input
        if out.max() > 1:
            out = out / 255
            # out *= 2
            # out -= 1
        for conv in self.convs:
            out = F.elu(conv(out))
        out = out.view(input.shape[0], -1)
        out = self.fc(out)
        out = self.ln(out)
        out = F.tanh(out)
        return out

# class ImageEncoder(nn.Module):
#
#     def __init__(self, state_dim, state_embed_dim, hidden_sizes):
#         super(ImageEncoder, self).__init__()
#         self.state_dim = state_dim
#         self.state_embed_dim = state_embed_dim
#         self.activation_func = nn.ReLU(inplace=True)
#         self.hidden1 = nn.Linear(state_dim[0]*state_dim[1]*state_dim[2], 1024)
#         self.hidden2 = nn.Linear(1024, state_embed_dim)
#
#
#     def forward(self, input):
#         input = input.view(-1, self.state_dim[0]*self.state_dim[1]*self.state_dim[2])
#         out = input
#         if out.max() > 1:
#             out = out / 255
#         out = self.activation_func(self.hidden1(out))
#         out = self.activation_func(self.hidden2(out))
#         return out
