# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ast import arg
import numpy as np
import random

from bidexhands.utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_sarl import process_sarl
from bidexhands.utils.process_marl import process_MultiAgentRL, get_AgentIndex
from bidexhands.utils.process_mtrl import *
from bidexhands.utils.process_metarl import *
from bidexhands.utils.process_offrl import *
from bidexhands.dexpoint.real_world import task_setting

MARL_ALGOS = ["mappo", "happo", "hatrpo","maddpg","ippo"]
SARL_ALGOS = ["ppo","ddpg","sac","td3","trpo"]
MTRL_ALGOS = ["mtppo", "random"]
META_ALGOS = ["mamlppo"]
OFFRL_ALGOS = ["td3_bc", "bcq", "iql", "ppo_collect"]

class DexPointProcessor:
    def __init__(self, cfg):
        self.obj_points = 1024
        self.hand_points = 512
        self.noise_std = 0.02

    def generate_point_cloud(self, depth_image, camera_intrinsics):
        height, width = depth_image.shape
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u = u.astype(np.float32)
        v = v.astype(np.float32)
        z = depth_image.astype(np.float32)
        if np.min(z) < 0:
            z = np.maximum(z, 0)
        
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, points.shape)
            points += noise

        return points


    def downsample_point_cloud(self, points, num_points):
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float32)
        
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
        elif len(points) < num_points:
            indices = np.random.choice(len(points), num_points - len(points), replace=True)
            points = np.concatenate([points, points[indices]], axis=0)
        return points

def train():
    print("Algorithm: ", args.algo)
    agent_index = get_AgentIndex(cfg)
    assert args.algo in MARL_ALGOS + SARL_ALGOS + MTRL_ALGOS + META_ALGOS + OFFRL_ALGOS, \
        "Unrecognized algorithm!\nAlgorithm should be one of: [happo, hatrpo, mappo,ippo, \
            maddpg,sac,td3,trpo,ppo,ddpg, mtppo, random, mamlppo, td3_bc, bcq, iql, ppo_collect]"
    algo = args.algo
    if args.algo in MARL_ALGOS: 
        # maddpg exists a bug now 
        args.task_type = "MultiAgent"
        algo = "MultiAgentRL"
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
        runner = eval('process_{}'.format(algo))(args, env, cfg_train, args.model_dir)
        if args.model_dir != "":
            runner.eval(1000)
        else:
            runner.run()
        return
    elif args.algo in SARL_ALGOS:
        algo = "sarl"
    elif args.algo in MTRL_ALGOS:
        args.task_type = "MultiTask"
    elif args.algo in META_ALGOS:
        args.task_type = "Meta"
    elif args.algo in OFFRL_ALGOS:
        pass 

    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

    if args.task == "XarmAllegroHandOver":
        task.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])
        task.setup_visual_obs_config({
            "relocate-point_cloud": {
                "num_points": 512,
                "noise": 0.02,
                "radius": (0.15, 0.8),
                "use_color": False
            }
        })
        task.pc_processor = DexPointProcessor(cfg)
    
    runner = eval('process_{}'.format(algo))(args, env, cfg_train, logdir)
    iterations = cfg_train["learn"]["max_iterations"]
    if args.max_iterations > 0:
        iterations = args.max_iterations

    runner.train(train_epoch=iterations) if args.algo in META_ALGOS else \
        runner.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])
        
if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    train()
