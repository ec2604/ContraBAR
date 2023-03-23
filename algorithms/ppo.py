import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import kornia
from models.policy import init
from utils import helpers as utl
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO:
    def __init__(self, args, actor_critic, value_loss_coef, entropy_coef, policy_optimiser, policy_anneal_lr,
                 train_steps, optimizer_representation_learner=None, lr=None, clip_param=0.2, ppo_epoch=5,
                 num_mini_batch=5, eps=None, use_huber_loss=True, use_clipped_value_loss=True, **kwargs):
        self.args = args

        # the model
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_huber_loss = use_huber_loss

        # optimiser
        if policy_optimiser == 'adam':
            self.optimiser = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        elif policy_optimiser == 'rmsprop':
            self.optimiser = optim.RMSprop(actor_critic.parameters(), lr=lr, eps=eps, alpha=0.99)
        self.optimizer_representation_learner = optimizer_representation_learner

        self.lr_scheduler_policy = None
        self.lr_scheduler_encoder = None
        if policy_anneal_lr:
            lam = lambda f: 1 - f / train_steps
            self.lr_scheduler_policy = optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda=lam)
    # def reset(self):
    #     init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
    #                    nn.init.calculate_gain('tanh'))
    #     self.optimiser = optim.Adam(self.actor_critic.parameters(), lr=self.args.lr_policy, eps=self.args.policy_eps)
    #     self.actor_critic.actor_layers[-1] = init_(self.actor_critic.actor_layers[-1])
    #     self.actor_critic.critic_layers[-1] = init_(self.actor_critic.critic_layers[-1])
    #     self.actor_critic.critic_linear = torch.nn.Linear(self.args.policy_layers[-1], 1).to(device)
    def policy_update(self, policy_storage):

        # -- get action values --
        advantages = policy_storage.returns[:-1] - policy_storage.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)



        # update the normalisation parameters of policy inputs before updating
        self.actor_critic.update_rms(args=self.args, policy_storage=policy_storage)

        # call this to make sure that the action_log_probs are computed
        # (needs to be done right here because of some caching thing when normalising actions)
        policy_storage.before_update(self.actor_critic)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        clip_frac_epoch = 0
        loss_epoch = 0
        for e in range(self.ppo_epoch):

            data_generator = policy_storage.feed_forward_generator(advantages, self.num_mini_batch)
            for sample in data_generator:

                state_batch,  task_batch, \
                actions_batch, hidden_batch, value_preds_batch, \
                return_batch, old_action_log_probs_batch, adv_targ = sample
                state_batch = state_batch.detach()
                hidden_batch = hidden_batch.detach()


                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = \
                    self.actor_critic.evaluate_actions(state=state_batch, latent=hidden_batch, task=task_batch,
                                                       action=actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
                clip_frac = torch.greater(torch.abs(ratio -1),self.clip_param).float().mean().item()
                if self.use_huber_loss and self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = F.smooth_l1_loss(values, return_batch, reduction='none')
                    value_losses_clipped = F.smooth_l1_loss(value_pred_clipped, return_batch, reduction='none')
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                elif self.use_huber_loss:
                    value_loss = F.smooth_l1_loss(values, return_batch)
                elif self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                if self.args.augment_for_policy:
                    obs_list = list(torch.split(state_batch, dim=1, split_size_or_sections=[3, state_batch.shape[1]-3]))
                    orig_shape = obs_list[0].shape
                    rgb = obs_list[0] / 255
                    rgb = kornia.augmentation.RandomRGBShift(p=1.)(rgb) * 255
                    # rgb = kornia.augmentation.RandomPlasmaShadow(p=1., shade_intensity=(-0.2, 0))(rgb) * 255
                    obs_list[0] = rgb
                    aug_state_batch = torch.cat(obs_list, dim=1)

                    _, new_actions = self.actor_critic.act(state=aug_state_batch, latent=hidden_batch, task=task_batch)
                    values_aug, action_log_probs_aug, dist_entropy_aug  = \
                        self.actor_critic.evaluate_actions(aug_state_batch, \
                                                       hidden_batch, task_batch, new_actions)
                    # Compute Augmented Loss
                    action_loss_aug = - action_log_probs_aug.mean()
                    value_loss_aug = .5 * (torch.detach(values) - values_aug).pow(2).mean()
                    aug_loss = value_loss_aug + action_loss_aug

                # zero out the gradients
                self.optimiser.zero_grad()


                # compute policy loss and backprop
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                if self.args.augment_for_policy:
                    loss = loss + 0.15* aug_loss

                # compute gradients (will attach to all networks involved in this computation)
                loss.backward()

                # clip gradients
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.policy_max_grad_norm)


                # update
                self.optimiser.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                loss_epoch += loss.item()
                clip_frac_epoch += clip_frac



        if self.lr_scheduler_policy is not None:
            self.lr_scheduler_policy.step()
        if self.lr_scheduler_encoder is not None:
            self.lr_scheduler_encoder.step()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        loss_epoch /= num_updates
        clip_frac_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, loss_epoch, clip_frac_epoch

    def act(self, state, latent, task, deterministic=False):
        return self.actor_critic.act(state=state, latent=latent, task=task, deterministic=deterministic)
