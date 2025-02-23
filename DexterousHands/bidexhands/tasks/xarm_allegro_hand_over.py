# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from matplotlib.pyplot import axis
import numpy as np
import os
import random
import torch

from bidexhands.utils.torch_jit_utils import *
from bidexhands.tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi


class XarmAllegroHandOver(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index

        self.is_multi_agent = is_multi_agent

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.allegro_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg", "pen", "ycb/banana", "ycb/can", "ycb/mug", "ycb/brick"]

        self.ignore_z = (self.object_type == "egg")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            "ycb/banana": "urdf/ycb/011_banana/011_banana.urdf",
            "ycb/can": "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf",
            "ycb/mug": "urdf/ycb/025_mug/025_mug.urdf",
            "ycb/brick": "urdf/ycb/061_foam_brick/061_foam_brick.urdf"
        }

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["openai", "full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.num_obs_dict = {
            "openai": 42,
            "full_no_vel": 77,
            "full": 157,
            "full_state": 390
        }
        self.num_hand_obs = 72 + 95 + 22
        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        if self.is_multi_agent:
            self.num_agents = 2
            self.cfg["env"]["numActions"] = 22
            
        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 44

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.allegro_hand_default_dof_pos = torch.zeros(self.num_allegro_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.allegro_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_allegro_hand_dofs]
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0]
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1]

        self.allegro_hand_another_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_allegro_hand_dofs:self.num_allegro_hand_dofs*2]
        self.allegro_hand_another_dof_pos = self.allegro_hand_another_dof_state[..., 0]
        self.allegro_hand_another_dof_vel = self.allegro_hand_another_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        self.cameras = []
        self.obs_cfg = None
        self.contact_reward_scale = self.cfg["env"].get("contactRewardScale", 0.5)
        self.lift_reward_scale = self.cfg["env"].get("liftRewardScale", 1.0)
    

    def get_depth_image(self):
        if not self.cameras:
            raise ValueError("未配置相机")
        
        camera_handle = self.cameras[0]
        depth_image = self.gym.get_camera_image(self.sim, self.envs[0], camera_handle, gymapi.IMAGE_DEPTH)

        if isinstance(depth_image, np.ndarray) and depth_image.size == 0:
            self.gym.step_graphics(self.sim)
            depth_image = self.gym.get_camera_image(
                self.sim, self.envs[0], camera_handle, 
                gymapi.IMAGE_DEPTH).astype(np.float32)
        
        if depth_image.shape[0] == 0 or depth_image.shape[1] == 0:
            depth_image = np.random.rand(480, 640) * 0.5
        
        return depth_image

    
    def setup_camera_from_config(self, camera_config):
        import sapien.core as sapien
        
        cam_params = camera_config["relocate"]
        cam_props = gymapi.CameraProperties()
        cam_props.width = 640
        cam_props.height = 480
        cam_props.horizontal_fov = 75.0
        
        gym_pos = gymapi.Vec3(0.5, 0.5, 0.8)
        gym_rot = gymapi.Quat.from_euler_zyx(-np.pi/2, np.pi/6, 0)
        
        cam_handle = self.gym.create_camera_sensor(self.envs[0], cam_props)
        self.gym.set_camera_transform(cam_handle, self.envs[0], 
                                    gymapi.Transform(gym_pos, gym_rot))
        self.cameras.append(cam_handle)
        
        print(f"[FIXED] 相机位置固定: pos={gym_pos} rotation={gym_rot}")
        print(f"View matrix:\n{self.gym.get_camera_view_matrix(self.sim, self.envs[0], cam_handle)}")


    
    def setup_visual_obs_config(self, obs_cfg):
        self.obs_cfg = obs_cfg
        self.visual_obs_keys = []
        
        original_obs_dim = self.cfg["env"]["numObservations"]
        
        if "relocate-point_cloud" in obs_cfg:
            pc_cfg = obs_cfg["relocate-point_cloud"]
            self.visual_obs_keys.append("relocate-point_cloud")
            
            # 配置点云参数
            num_points = pc_cfg.get("num_points", 512)
            self.point_cloud_kwargs = {
                'num_points': num_points,
                'noise': pc_cfg.get("noise", 0.02),
                'radius': pc_cfg.get("radius", (0.2, 0.7))
            }
            
            self.cfg["env"]["numObservations"] += num_points * 3
            self.pc_obs_start_idx = original_obs_dim
            
            # 配置相机参数
            self.camera_intrinsics = np.array([
                [500, 0, 320],  # fx, cx
                [0, 500, 240],  # fy, cy
                [0, 0, 1]
            ])


    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        print(f"Creating {num_envs} environments")
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../assets"
        allegro_hand_asset_file = "urdf/xarm_description/urdf/xarm6.urdf"
        allegro_hand_another_asset_file = "urdf/xarm_description/urdf/xarm6.urdf"

        object_asset_file = self.asset_files_dict[self.object_type]

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.use_mesh_materials = True  # 启用材质支持
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        # 加载机械臂资产 (含STL引用)
        allegro_hand_asset = self.gym.load_asset(
            self.sim, 
            asset_root, 
            allegro_hand_asset_file, 
            asset_options
        )

        # 获取手部各关键刚体索引
        self.thumb_body_idx = self.gym.find_asset_rigid_body_index(
            allegro_hand_asset, 
            "lf5"  # 拇指末端link名称为lf5
        )
        self.index_body_idx = self.gym.find_asset_rigid_body_index(
            allegro_hand_asset, 
            "if5"  # 食指末端link名称
        )
        self.middle_body_idx = self.gym.find_asset_rigid_body_index(
            allegro_hand_asset,
            "mf5"  # 中指末端link名称 
        )
        self.ring_body_idx = self.gym.find_asset_rigid_body_index(
            allegro_hand_asset, 
            "pf5"  # 小指末端link
        )

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        allegro_hand_asset = self.gym.load_asset(self.sim, asset_root, allegro_hand_asset_file, asset_options)
        allegro_hand_another_asset = self.gym.load_asset(self.sim, asset_root, allegro_hand_another_asset_file, asset_options)

        self.num_allegro_hand_bodies = self.gym.get_asset_rigid_body_count(allegro_hand_asset)
        self.num_allegro_hand_shapes = self.gym.get_asset_rigid_shape_count(allegro_hand_asset)
        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(allegro_hand_asset)
        self.num_allegro_hand_actuators = self.gym.get_asset_dof_count(allegro_hand_asset)
        self.num_allegro_hand_tendons = self.gym.get_asset_tendon_count(allegro_hand_asset)
        self.num_allegro_hand_dof_dicts = self.gym.get_asset_dof_dict(allegro_hand_asset)

        print("self.num_allegro_hand_bodies: ", self.num_allegro_hand_bodies)
        print("self.num_allegro_hand_shapes: ", self.num_allegro_hand_shapes)
        print("self.num_allegro_hand_dofs: ", self.num_allegro_hand_dofs)
        print("self.num_allegro_hand_actuators: ", self.num_allegro_hand_actuators)
        print("self.num_allegro_hand_tendons: ", self.num_allegro_hand_tendons)
        print("self.num_allegro_hand_dof_dicts: ", self.num_allegro_hand_dof_dicts)

        # tendon set up
        limit_stiffness = 3
        t_damping = 0.1
        tendon_props = self.gym.get_asset_tendon_properties(allegro_hand_asset)
        a_tendon_props = self.gym.get_asset_tendon_properties(allegro_hand_another_asset)

        for i in range(self.num_allegro_hand_tendons):
            tendon_props[i].limit_stiffness = limit_stiffness
            tendon_props[i].damping = t_damping

        self.gym.set_asset_tendon_properties(allegro_hand_asset, tendon_props)
        self.gym.set_asset_tendon_properties(allegro_hand_another_asset, a_tendon_props)
        
        self.actuated_dof_indices = [i for i in range(self.num_allegro_hand_dofs)]

        # set allegro_hand dof properties
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(allegro_hand_asset)
        allegro_hand_another_dof_props = self.gym.get_asset_dof_properties(allegro_hand_another_asset)

        self.allegro_hand_dof_lower_limits = []
        self.allegro_hand_dof_upper_limits = []
        self.allegro_hand_dof_default_pos = []
        self.allegro_hand_dof_default_vel = []
        self.allegro_hand_dof_stiffness = []
        self.allegro_hand_dof_damping = []
        self.allegro_hand_dof_effort = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        for i in range(self.num_allegro_hand_dofs):
            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props['upper'][i])
            self.allegro_hand_dof_default_pos.append(0.0)
            self.allegro_hand_dof_default_vel.append(0.0)
        for i in range(self.num_allegro_hand_dofs):
            allegro_hand_dof_props['stiffness'][i] = 3
            allegro_hand_dof_props['damping'][i] = 0.1
            allegro_hand_dof_props['effort'][i] = 0.5
            allegro_hand_another_dof_props['stiffness'][i] = 3
            allegro_hand_another_dof_props['damping'][i] = 0.1
            allegro_hand_another_dof_props['effort'][i] = 0.5

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.allegro_hand_dof_lower_limits = to_torch(self.allegro_hand_dof_lower_limits, device=self.device)
        self.allegro_hand_dof_upper_limits = to_torch(self.allegro_hand_dof_upper_limits, device=self.device)
        self.allegro_hand_dof_default_pos = to_torch(self.allegro_hand_dof_default_pos, device=self.device)
        self.allegro_hand_dof_default_vel = to_torch(self.allegro_hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        allegro_hand_start_pose = gymapi.Transform()
        allegro_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.5, self.up_axis_idx))
        allegro_hand_start_pose.r = gymapi.Quat().from_euler_zyx(1.571, -1.571, 0)

        allegro_another_hand_start_pose = gymapi.Transform()
        allegro_another_hand_start_pose.p = gymapi.Vec3(0, -0.4, 0.5)
        allegro_another_hand_start_pose.r = gymapi.Quat().from_euler_zyx(-1.571, -1.571, 0)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = allegro_hand_start_pose.p.x
        pose_dy, pose_dz = -0.1, 0.06

        object_start_pose.p.y = allegro_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = allegro_hand_start_pose.p.z + pose_dz

        if self.object_type == "pen":
            object_start_pose.p.z = allegro_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.0

        # compute aggregate size
        max_agg_bodies = self.num_allegro_hand_bodies * 2 + 2
        max_agg_shapes = self.num_allegro_hand_shapes * 2 + 2

        self.allegro_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.another_hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            allegro_hand_actor = self.gym.create_actor(env_ptr, allegro_hand_asset, allegro_hand_start_pose, "hand", i, -1, 0)
            allegro_hand_another_actor = self.gym.create_actor(env_ptr, allegro_hand_another_asset, allegro_another_hand_start_pose, "another_hand", i, -1, 0)
            
            self.hand_start_states.append([allegro_hand_start_pose.p.x, allegro_hand_start_pose.p.y, allegro_hand_start_pose.p.z,
                                           allegro_hand_start_pose.r.x, allegro_hand_start_pose.r.y, allegro_hand_start_pose.r.z, allegro_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            
            self.gym.set_actor_dof_properties(env_ptr, allegro_hand_actor, allegro_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, allegro_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            self.gym.set_actor_dof_properties(env_ptr, allegro_hand_another_actor, allegro_hand_another_dof_props)
            another_hand_idx = self.gym.get_actor_index(env_ptr, allegro_hand_another_actor, gymapi.DOMAIN_SIM)
            self.another_hand_indices.append(another_hand_idx)            

            # randomize colors and textures for rigid body
            num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, allegro_hand_actor)
            hand_rigid_body_index = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]

            # create fingertip force-torque sensors
            if self.obs_type == "full_state" or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, allegro_hand_actor)
                self.gym.enable_actor_dof_force_sensors(env_ptr, allegro_hand_another_actor)
            
            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.allegro_hands.append(allegro_hand_actor)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        # self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.another_hand_indices = to_torch(self.another_hand_indices, dtype=torch.long, device=self.device)

        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "egg")
        )

        # 新增的论文奖励组件 -----------------------------------------------
        # 获取接触力数据（需在step前刷新）
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, -1, 3)
        
        # 接触奖励计算（论文式2）
        thumb_contact = torch.norm(self.contact_forces[:, self.thumb_body_idx], dim=1) > 0.1
        finger_contacts = sum(
            torch.norm(self.contact_forces[:, idx], dim=1) > 0.1 
            for idx in [self.index_body_idx, self.middle_body_idx, self.ring_body_idx]
        )
        contact_reward = (thumb_contact & (finger_contacts >= 2)).float() * self.contact_reward_scale

        # 举升奖励计算
        current_height = self.object_pos[:, 2]
        target_height = self.goal_pos[:, 2]
        lift_reward = (current_height >= target_height * 0.8).float() * self.lift_reward_scale  # 达到目标高度的80%

        # 合并奖励（论文式4）
        self.rew_buf += contact_reward + lift_reward
        # ----------------------------------------------------------------

        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))
        return self.rew_buf

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        # 获取深度图
        depth_image = self.get_depth_image()
        real_point_cloud = self.pc_processor.generate_point_cloud(depth_image, self.camera_intrinsics)
        real_point_cloud = self.pc_processor.downsample_point_cloud(real_point_cloud, self.point_cloud_kwargs['num_points'])

        if real_point_cloud.size == 0:
            real_point_cloud = np.random.rand(512, 3).astype(np.float32)

        point_cloud_tensor = torch.from_numpy(real_point_cloud.astype(np.float32)).cuda(device=self.obs_buf.device)
        point_cloud_flat = point_cloud_tensor.flatten()

        obs_start_idx = self.pc_obs_start_idx
        if point_cloud_flat.size(0) != self.obs_buf.shape[1] - obs_start_idx:
            if point_cloud_flat.size(0) > self.obs_buf.shape[1] - obs_start_idx:
                point_cloud_flat = point_cloud_flat[:self.obs_buf.shape[1] - obs_start_idx]
            else:
                padding = torch.zeros(self.obs_buf.shape[1] - obs_start_idx - point_cloud_flat.size(0), device=self.obs_buf.device)
                point_cloud_flat = torch.cat([point_cloud_flat, padding], dim=0)

        self.obs_buf[:, obs_start_idx:obs_start_idx + point_cloud_flat.size(0)] = point_cloud_flat

        
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.compute_full_state()

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_state(self, asymm_obs=False):
        # fingertip observations, state(pose and vel) + force-torque sensors
        # num_ft_states = 13 * int(self.num_fingertips / 2)  # 65
        # num_ft_force_torques = 6 * int(self.num_fingertips / 2)  # 30

        self.obs_buf[:, 0:self.num_allegro_hand_dofs] = unscale(self.allegro_hand_dof_pos,
                                                            self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        self.obs_buf[:, self.num_allegro_hand_dofs:2*self.num_allegro_hand_dofs] = self.vel_obs_scale * self.allegro_hand_dof_vel
        fingertip_obs_start = 72  # 168 = 157 + 11

        action_obs_start = fingertip_obs_start + 95
        self.obs_buf[:, action_obs_start:action_obs_start + 22] = self.actions[:, :22]

        # another_hand
        another_hand_start = action_obs_start + 16
        self.obs_buf[:, another_hand_start:self.num_allegro_hand_dofs + another_hand_start] = unscale(self.allegro_hand_another_dof_pos,
                                                            self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        self.obs_buf[:, self.num_allegro_hand_dofs + another_hand_start:2*self.num_allegro_hand_dofs + another_hand_start] = self.vel_obs_scale * self.allegro_hand_another_dof_vel

        fingertip_another_obs_start = another_hand_start + 72

        action_another_obs_start = fingertip_another_obs_start + 95
        self.obs_buf[:, action_another_obs_start:action_another_obs_start + 22] = self.actions[:, 22:]

        obj_obs_start = action_another_obs_start + 16  # 144
        self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

        goal_obs_start = obj_obs_start + 13  # 157 = 144 + 13
        self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
        self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))


    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 1] -= 0.25
        self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset(self, env_ids, goal_env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_allegro_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        if self.object_type == "pen":
            rand_angle_y = torch.tensor(0.3)
            new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
                                                    self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids], self.z_unit_tensor[env_ids])

        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))

        # reset shadow hand
        delta_max = self.allegro_hand_dof_upper_limits - self.allegro_hand_dof_default_pos
        delta_min = self.allegro_hand_dof_lower_limits - self.allegro_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5+self.num_allegro_hand_dofs]

        pos = self.allegro_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta

        self.allegro_hand_dof_pos[env_ids, :] = pos
        self.allegro_hand_another_dof_pos[env_ids, :] = pos

        self.allegro_hand_dof_vel[env_ids, :] = self.allegro_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_allegro_hand_dofs:5+self.num_allegro_hand_dofs*2]   

        self.allegro_hand_another_dof_vel[env_ids, :] = self.allegro_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_allegro_hand_dofs:5+self.num_allegro_hand_dofs*2]

        self.prev_targets[env_ids, :self.num_allegro_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_allegro_hand_dofs] = pos

        self.prev_targets[env_ids, self.num_allegro_hand_dofs:self.num_allegro_hand_dofs*2] = pos
        self.cur_targets[env_ids, self.num_allegro_hand_dofs:self.num_allegro_hand_dofs*2] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        another_hand_indices = self.another_hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.unique(torch.cat([hand_indices,
                                                 another_hand_indices]).to(torch.int32))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))  

        all_indices = torch.unique(torch.cat([all_hand_indices,
                                                 object_indices]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.allegro_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.allegro_hand_dof_lower_limits[self.actuated_dof_indices], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, :22],
                                                                   self.allegro_hand_dof_lower_limits[self.actuated_dof_indices], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.allegro_hand_dof_lower_limits[self.actuated_dof_indices], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices])

            self.cur_targets[:, self.actuated_dof_indices + 22] = scale(self.actions[:, 22:44],
                                                                   self.allegro_hand_dof_lower_limits[self.actuated_dof_indices], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices + 22] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices + 22] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices + 22] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices + 22],
                                                                          self.allegro_hand_dof_lower_limits[self.actuated_dof_indices], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices])
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.prev_targets[:, self.actuated_dof_indices + 22] = self.cur_targets[:, self.actuated_dof_indices + 22]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                targetx = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                targety = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                targetz = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.goal_pos[i].cpu().numpy() + self.goal_displacement_tensor.cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])

                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist
    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(goal_dist) <= 0, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threashold
    reward = torch.where(object_pos[:, 2] <= 0.2, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(object_pos[:, 2] <= 0.2, torch.ones_like(reset_buf), reset_buf)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot
