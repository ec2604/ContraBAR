import argparse
from config.dm_control import args_reacher_varibad, args_sparse_reacher_varibad
from config.pointrobot import args_pointrobot_image_varibad
from config.mujoco import args_sparse_ant_goal_varibad
from config.panda_gym import args_sparse_panda_reacher_varibad
from environments.parallel_envs import make_vec_envs
import pickle
import torch
import numpy as np
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args, rest_args = parser.parse_known_args()
    args = args_sparse_panda_reacher_varibad.get_args(rest_args)
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_90__16:09_21:25:28/'
    # model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_16__23:09_20:19:52/'
    model_location = '/mnt/data/erac/logs_CustomReach-v0/contrabar_21__28:09_20:57:43/'


    with open(model_location + 'models/env_rew_rms.pkl', 'rb') as f:
        a = pickle.load(f)
    env = make_vec_envs(env_name='CustomReach-v0',
                        seed=12 * 42 +10000,
                        num_processes=1,
                        gamma=0.99,
                        device=torch.device('cpu'),
                        episodes_per_task=3,
                        normalise_rew=True, ret_rms=a,
                        rank_offset=16 + 42,  # not sure if the temp folders would otherwise clash
                        tasks=None
                        )
    encoder = torch.load(model_location + 'models/encoder.pt')
    policy = torch.load(model_location + 'models/policy.pt')
    # belief_evaluator = torch.load("/mnt/data/erac/logs_CustomReach-v0/belief_panda_seed_16.pt")
    unwrapped_env = env.venv.unwrapped.envs[0]
    #goal_pos = np.array([(np.pi)/2, 0.2])
    #start_pos = np.array([0.])
    #unwrapped_env.visualise_behaviour_for_script(env, args,policy, 100, encoder, image_folder='/mnt/data/erac/logs_SparseReacher-v0/test',goal_pos=goal_pos,start_pos=start_pos,belief_evaluator=None)
    unwrapped_env.visualise_behaviour(env, args, policy, 100, encoder,image_folder='/mnt/data/erac/logs_CustomReach-v0/test',belief_evaluator=None)#belief_evaluator)


# import isaacgym
# import isaacgymenvs
# import torch
# import numpy as np
# from isaacgym import gymapi, gymtorch
# cams = []
# cam_tensors = []
# envs = isaacgymenvs.make(seed=0,
#                          task="FrankaCubeStack",
#                          num_envs=100,
#                          sim_device="cuda:2",
#                          rl_device="cuda:2",
#                          graphics_device_id=2,
#                          headless=True)
# envs.reset()
# for env in envs.envs:
#     cam_props = gymapi.CameraProperties()
#     cam_props.width = 128
#     cam_props.height = 128
#     cam_props.enable_tensors = True
#     cam_handle = envs.gym.create_camera_sensor(env, cam_props)
#     envs.gym.set_camera_location(cam_handle, env, gymapi.Vec3(1., 0., 2.),
#                                  gymapi.Vec3(0, 0., 1.))
#     cams.append(cam_handle)
#     cam_tensor = envs.gym.get_camera_image_gpu_tensor(envs.sim, env, cam_handle, gymapi.ImageType.IMAGE_COLOR)
#     torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
#     cam_tensors.append(torch_cam_tensor)
# envs.gym.prepare_sim(envs.sim)
# envs.gym.simulate(envs.sim)
# envs.gym.fetch_results(envs.sim, True)
# envs.gym.step_graphics(envs.sim)
# envs.gym.render_all_camera_sensors(envs.sim)
# envs.gym.start_access_image_tensors(envs.sim)
