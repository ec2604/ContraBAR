import argparse
from utils.helpers import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    # --- GENERAL ---

    # training parameters
    parser.add_argument('--num_frames', type=int, default=5e7, help='number of frames to train')
    parser.add_argument('--max_rollouts_per_task', type=int, default=3)
    parser.add_argument('--exp_label', default='contrabar', help='label (typically name of method)')
    parser.add_argument('--env_name', default='PointEnvWind-v0', help='environment to train on')

    # --- POLICY ---

    # what to pass to the policy (note this is after the encoder)
    parser.add_argument('--pass_state_to_policy', type=boolean_argument, default=True, help='condition policy on state')
    parser.add_argument('--pass_latent_to_policy', type=boolean_argument, default=True, help='condition policy on VAE latent')
    parser.add_argument('--from_pixels', type=boolean_argument, default=False, help='Whether state input to policy is pixel-based')
    parser.add_argument('--pass_task_to_policy', type=boolean_argument, default=False, help='condition policy on ground-truth task description')

    # using separate encoders for the different inputs ("None" uses no encoder)
    parser.add_argument('--policy_state_embedding_dim', type=int, default=64)
    parser.add_argument('--policy_latent_embedding_dim', type=int, default=None)
    parser.add_argument('--policy_belief_embedding_dim', type=int, default=None)
    parser.add_argument('--policy_task_embedding_dim', type=int, default=None)

    # normalising (inputs/rewards/outputs)
    parser.add_argument('--norm_state_for_policy', type=boolean_argument, default=True, help='normalise state input')
    parser.add_argument('--norm_latent_for_policy', type=boolean_argument, default=False, help='normalise latent input')
    parser.add_argument('--norm_task_for_policy', type=boolean_argument, default=True, help='normalise task input')
    parser.add_argument('--norm_rew_for_policy', type=boolean_argument, default=True, help='normalise rew for RL train')
    parser.add_argument('--norm_actions_pre_sampling', type=boolean_argument, default=False, help='normalise policy output')
    parser.add_argument('--norm_actions_post_sampling', type=boolean_argument, default=True, help='normalise policy output')
    parser.add_argument('--transform_state_to_latent', type=boolean_argument, default=False,
                        help='transform state to encoded state')
    # network
    parser.add_argument('--policy_layers', nargs='+', default=[512, 512])
    parser.add_argument('--policy_activation_function', type=str, default='tanh', help='tanh/relu/leaky-relu')
    parser.add_argument('--policy_initialisation', type=str, default='orthogonal', help='normc/orthogonal')
    parser.add_argument('--policy_anneal_lr', type=boolean_argument, default=False, help='anneal LR over time')

    # RL algorithm
    parser.add_argument('--policy', type=str, default='ppo', help='choose: a2c, ppo')
    parser.add_argument('--policy_optimiser', type=str, default='adam', help='choose: rmsprop, adam')

    # PPO specific
    parser.add_argument('--ppo_num_epochs', type=int, default=2, help='number of epochs per PPO update')
    parser.add_argument('--ppo_num_minibatch', type=int, default=2, help='number of minibatches to split the data')
    parser.add_argument('--ppo_use_huberloss', type=boolean_argument, default=True, help='use huberloss instead of MSE')
    parser.add_argument('--ppo_use_clipped_value_loss', type=boolean_argument, default=True, help='clip value loss')
    parser.add_argument('--ppo_clip_param', type=float, default=0.1, help='clamp param')

    # other hyperparameters
    parser.add_argument('--lr_policy', type=float, default=1e-4, help='learning rate (default: 7e-4)')
    parser.add_argument('--num_processes', type=int, default=16,
                        help='how many training CPU processes / parallel environments to use (default: 16)')
    parser.add_argument('--policy_num_steps', type=int, default=150,
                        help='number of env steps to do (per process) before updating')
    parser.add_argument('--policy_eps', type=float, default=1e-8, help='optimizer epsilon (1e-8 for ppo, 1e-5 for a2c)')
    parser.add_argument('--policy_init_std', type=float, default=1.0, help='only used for continuous actions')
    parser.add_argument('--policy_value_loss_coef', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument('--policy_entropy_coef', type=float, default=0.01, help='entropy term coefficient')
    parser.add_argument('--policy_gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--policy_use_gae', type=boolean_argument, default=True,
                        help='use generalized advantage estimation')
    parser.add_argument('--policy_tau', type=float, default=0.9, help='gae parameter')
    parser.add_argument('--use_proper_time_limits', type=boolean_argument, default=True,
                        help='treat timeout and death differently (important in mujoco)')
    parser.add_argument('--policy_max_grad_norm', type=float, default=0.5, help='max norm of gradients')
    parser.add_argument('--encoder_max_grad_norm', type=float, default=None, help='max norm of gradients')
    parser.add_argument('--decoder_max_grad_norm', type=float, default=None, help='max norm of gradients')

    # --- REP LEARNER TRAINING ---

    # cpc
    parser.add_argument('--negative_factor', type=int, default=16, help='number of negative samples per positive for CPC batch')
    parser.add_argument('--sampling_method', type=str, default='fast', help='fast/precise/negative_rewards')
    # general
    parser.add_argument('--with_action_gru', type=boolean_argument, default=False, help='include action_gru to contrast beliefs')
    parser.add_argument('--lr_representation_learner', type=float, default=1e-4)
    parser.add_argument('--num_trajs_representation_learning_buffer', type=int, default=500,
                        help='how many trajectories (!) to keep in VAE buffer')
    parser.add_argument('--underlying_state_dim', default=())

    parser.add_argument('--augment_z', type=bool, default=False, help='weight trajectory steps?')
    parser.add_argument('--precollect_len', type=int, default=12000,
                        help='how many trajectories to pre-collect before training begins (useful to fill VAE buffer)')
    parser.add_argument('--representation_learner_buffer_add_thresh', type=float, default=1,
                        help='probability of adding a new trajectory to buffer')
    parser.add_argument('--representation_learner_batch_num_trajs', type=int, default=20,
                        help='how many trajectories to use for CPC update')
    parser.add_argument('--tbptt_stepsize', type=int, default=None,
                        help='stepsize for truncated backpropagation through time; None uses max (horizon of BAMDP)')


    parser.add_argument('--num_representation_learner_updates', type=int, default=10,
                        help='how many CPC update steps to take per meta-iteration')
    parser.add_argument('--pretrain_len', type=int, default=0, help='for how many updates to pre-train the VAE')
    parser.add_argument('--evaluate_representation', type=boolean_argument, default=False, help='train MLP to evaluate latent, without gradients flowing back')
    parser.add_argument('--evaluator_lr', type=float, default=1e-3, help='lr for MLP to evaluate representation')
# - encoder
    parser.add_argument('--action_embedding_size', type=int, default=16)
    parser.add_argument('--state_embedding_size', type=int, default=16)
    parser.add_argument('--reward_embedding_size', type=int, default=16)
    parser.add_argument('--encoder_layers_before_gru', nargs='+', type=int, default=[])
    parser.add_argument('--encoder_gru_hidden_size', type=int, default=128, help='dimensionality of RNN hidden state')
    parser.add_argument('--encoder_layers_after_gru', nargs='+', type=int, default=[])
    parser.add_argument('--latent_dim', type=int, default=50, help='dimensionality of latent space')
    parser.add_argument('--lookahead_factor', type=int, default=10, help='lookahead for CPC')




    # for other things
    parser.add_argument('--disable_metalearner', type=boolean_argument, default=False,
                        help='Train feedforward policy')
    parser.add_argument('--single_task_mode', type=boolean_argument, default=False,
                        help='train policy on one (randomly chosen) environment only')

    # --- OTHERS ---

    # logging, saving, evaluation
    parser.add_argument('--log_interval', type=int, default=20, help='log interval, one log per n updates')
    parser.add_argument('--save_interval', type=int, default=500, help='save interval, one save per n updates')
    parser.add_argument('--save_intermediate_models', type=boolean_argument, default=False, help='save all models')
    parser.add_argument('--eval_interval', type=int, default=20, help='eval interval, one eval per n updates')
    parser.add_argument('--vis_interval', type=int, default=20, help='visualisation interval, one eval per n updates')
    parser.add_argument('--gradient_log_interval', type=int, default=100)
    parser.add_argument('--results_log_dir', default=None, help='directory to save results (None uses ./logs)')

    # general settings
    parser.add_argument('--seed',  nargs='+', type=int, default=[73])
    parser.add_argument('--deterministic_execution', type=boolean_argument, default=False,
                        help='Make code fully deterministic. Expects 1 process and uses deterministic CUDNN')

    return parser.parse_args(rest_args)
