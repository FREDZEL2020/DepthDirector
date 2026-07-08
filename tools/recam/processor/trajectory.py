import os
import json
import csv  
import torch
from processor.base import Warper
from data.monocular import Pi3X
import data.depth_model as DEPTH_MODELS
from geometry import Dynamic3DMesh
from utils.camutils import location_rotation_to_transform_matrix, generate_traj_specified, sphere2pose
import numpy as np
from utils.ioutils import read_video_frames, save_video
from imageio.v3 import imread, imwrite
from PIL import Image
import random
import time

class TrajectoryProcessor(Warper):
    def __init__(self, opts):
        super().__init__(opts)
        if len(opts.depth_model) > 0:
            print("USE DEPTH MODEL", opts.depth_model)
            self.depth_estimater = getattr(DEPTH_MODELS, opts.depth_model)(save_dir=self.save_dir)
        else:
            print("USE DEFAULT DEPTH MODEL") 
            self.depth_estimater = Pi3X(save_dir=self.save_dir)
        
        self.geometry = Dynamic3DMesh(opts.device)
        if hasattr(opts, 'geometry'):
            import geometry
            # 直接从geometry模块获取
            geometry_class = getattr(geometry, opts.geometry)
            self.geometry = geometry_class(opts.device)
        self.traj_choise = [
            "Orbit",
            "Helix_In",
            "S_Curve_Dolly",
            "Zoom-out",
            "Orbit_Ascend",
            "Parabola_Dive"
        ]
        # random.seed(0)

    def get_radius(self, depth_map):
        # depth_map: L,H,W,1
        assert len(depth_map.shape) == 4, depth_map.shape
        radius_scale = 1.1
        padding = 100
        # target_dis = depths[0, 0, depths.shape[-2] // 2 - padding : depths.shape[-2] // 2 + padding, depths.shape[-1] // 2 - padding : depths.shape[-1] // 2 + padding].cpu().mean()
        target_dis = depth_map[:,depth_map.shape[1] // 2 - padding : depth_map.shape[1] // 2 + padding, depth_map.shape[2] // 2 - padding : depth_map.shape[2] // 2 + padding].min()
        radius = (
            target_dis * radius_scale
        )
        print("radius: ",radius)
        return radius

    def get_poses(self, radius, num_frames, traj):
        """
        Docstring for get_poses
        
        :param self: Description
        :param radius: Description
        :param num_frames: Description
        :param traj: Description

        return torch.tensor
        """
        c2w_init = (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            .to(self.opts.device)
            .unsqueeze(0)
        )
        dtheta, dphi, dr, dx, dy = traj
        poses = generate_traj_specified(
            c2w_init, dtheta, dphi, dr * radius, dx * radius, dy * radius, num_frames, self.opts.device
        )
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        # poses = poses
        pose_s = poses[:1].repeat(num_frames, 1, 1)
        pose_t = poses
        #.flip(dims=[0])
        # pose_t = poses[-1:].repeat(num_frames, 1, 1)
        return pose_s, pose_t

    def get_poses_traj2traj(self, radius, num_frames, traj1, traj2):
        c2w_init = (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            .to(self.opts.device)
            .unsqueeze(0)
        )
        dtheta1, dphi1, dr1, dx1, dy1 = traj1
        dtheta2, dphi2, dr2, dx2, dy2 = traj2
        # Initialize a camera.
        thetas = np.linspace(dtheta1, dtheta2, num_frames)
        phis = np.linspace(dphi1, dphi2, num_frames)
        rs = np.linspace(dr1 * radius, dr2 * radius, num_frames)
        xs = np.linspace(dx1, dx2, num_frames)
        ys = np.linspace(dy1, dy2, num_frames)
        c2ws_list = []
        for th, ph, r, x, y in zip(thetas, phis, rs, xs, ys):
            c2w_new = sphere2pose(
                c2w_init,
                np.float32(th),
                np.float32(ph),
                np.float32(r),
                self.opts.device,
                np.float32(x),
                np.float32(y),
            )
            c2ws_list.append(c2w_new)
        poses = torch.cat(c2ws_list, dim=0)

        poses[:, 2, 3] = poses[:, 2, 3] + radius
        # poses = poses
        pose_s = poses[:1].repeat(num_frames, 1, 1)
        pose_t = poses
        #.flip(dims=[0])
        # pose_t = poses[-1:].repeat(num_frames, 1, 1)
        return pose_s, pose_t

    def warp_mono_video(self, input_info, save_dir, direction=('orbit',0,0,0,0,0), render_keys=None):
        opts = self.opts
        print("opts.video_length",self.opts.video_length)
        info = {}
        # N,H,W,C
        N,H,W = input_info['depth'].shape[:3]
        for key in input_info.keys():
            info[key] = input_info[key].astype(np.float32)
            # info[key] =  torch.asarray(input_info[key]).float()
        radius = self.get_radius(info['depth'])
        pose_s, pose_t = self.get_trajectory(radius, direction, opts.video_length)
        if 'K' not in info:
            fov = 90
            f = W / (2*np.tan(np.deg2rad(fov)/2))
            cx = W // 2
            cy = H // 2
            intrinsics = torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]]).repeat(N, 1, 1).to(self.opts.device)
            info['K'] = intrinsics
        print(info['K'][0])
        if 'pose' in info:
            # np.linalg.inv(gt[0]) @ gt
            pose = torch.asarray(info['pose']).float()
            cam_s = torch.linalg.inv(pose).to(pose_s.device)
            cam_s = torch.linalg.inv(pose_s[0]) @ (torch.linalg.inv(cam_s[0]) @ cam_s)
            pose_s = torch.linalg.inv(cam_s)
        os.makedirs(save_dir,exist_ok=True)
        self.geometry.build(frames=info['frames'], depths=info['depth'], Ks=info['K'], poses=pose_s)
        results = self.geometry.render(poses_t=pose_t, Ks=info['K'])
        if opts.save_depth:
            print("[Info] Save raw depth", f"{save_dir}/depth.npy")
            np.save(f"{save_dir}/depth.npy", info['depth'])
            for idx in results['extra']['meshes'].keys():
                for key in ["valid", "invalid", "dw"]:
                    results['extra']['meshes'][idx][key].export(f"{save_dir}/{key}_{idx}.ply")
        if render_keys is None:

            self.render_results(results, save_dir)
        else:
            self.render_results(results, save_dir, render_keys)
        # Add: Save Poses
        pose_info = {
            "K": info['K'][0].tolist(),
            # c2w
            "poses": torch.linalg.inv(pose_t).cpu().numpy().tolist()
        }
        with open(f"{save_dir}/cameras.json", "w") as f:
            json.dump(pose_info, f, indent=1)
        return f"{save_dir}/input.mp4"

    def process(self, override_video_path=None, override_save_dir=None):
        input_dir = override_video_path if override_video_path else self.opts.video_path
        save_dir = override_save_dir if override_save_dir else self.save_dir
        
        if self.opts.traj_txt is not None:
            traj_list = [self.opts.traj_txt]
        else:
            print("Use Predefined Trajectories")
            traj_list = [
                "Rotate,0,30,0,0,0",
                "Rotate,0,-30,0,0,0"
            ]
        frames=self.load_mono_video(input_dir, data_type="depthcrafter")
        original_frames=self.load_mono_video(input_dir, data_type="original")
        if frames.shape[0] < self.opts.video_length:
            frames = np.concatenate([frames, np.repeat(frames[-2:-1], self.opts.video_length - frames.shape[0], 0)], axis=0)
            original_frames = np.concatenate([original_frames, np.repeat(original_frames[-2:-1], self.opts.video_length - original_frames.shape[0], 0)], axis=0)
        os.makedirs(save_dir, exist_ok=True)
        self.depth_estimater.save_dir = save_dir
        results =  self.depth_estimater.video_depth(frames)

        for trajtxt in traj_list:
            if trajtxt=="Random":
                traj_type = random.choice(self.traj_choise)
                print("Random",traj_type)
                trajtxt = f"{traj_type},0,{random.choice([-1,1])},{random.choice([0.2,0.3,0.4])},0,0"
            traj = trajtxt.split(",")
            input_path = self.warp_mono_video(results, os.path.join(save_dir, trajtxt), traj)
            imwrite(input_path, (frames*255).astype(np.uint8), macro_block_size=1, fps=self.opts.fps)
            first_frame = (frames[0] * 255).astype(np.uint8)
            Image.fromarray(first_frame).save(os.path.join(os.path.dirname(input_path), "input.png"))

    def process_recursive(self):
        if self.opts.traj_txt is not None:
            traj_list = [self.opts.traj_txt]
        else:
            print("Use Predefined Trajectories")
            traj_list = [
            ]
        input_list = []
        video_list = []
        if os.path.isdir(self.opts.video_path):
            for root, dirs, files in os.walk(self.opts.video_path):
                # 检查当前目录下是否存在
                for file in files:
                    if file.endswith("mp4"):
                        input_list.append([os.path.join(root,file), f"{os.path.basename(root)}_{file.removesuffix('.mp4')}"])
        else:
            input_list.append([self.opts.video_path, self.opts.video_path.removesuffix('.mp4')])
        output_list = []
        for video_path, output_name in input_list:
            if os.path.exists(f"{self.save_dir}/{output_name}"):
                continue
            self.process(video_path, f"{self.save_dir}/{output_name}")

    def find_look_at_rotation(self, current_loc, current_focus_point, world_up):
        """
        向量化版本的相机旋转矩阵计算
        
        参数:
        -----------
        current_loc : np.ndarray, shape (N, 3)
            相机位置
        current_focus_point : np.ndarray, shape (N, 3)
            目标点
        
        返回:
        --------
        np.ndarray, shape (N, 3, 3)
            旋转矩阵
        """
        current_loc = np.asarray(current_loc)
        current_focus_point = np.asarray(current_focus_point)
        
        N = current_loc.shape[0]
        
        # 1. 计算前向向量
        forward = current_focus_point - current_loc
        forward_norm = np.linalg.norm(forward, axis=1, keepdims=True)
        forward = forward / (forward_norm + 1e-8)
        
        # 2. 默认上向量
        world_up = np.tile(world_up, (N, 1))
        
        # 3. 计算右向量
        right = -np.cross(world_up, forward)
        right_norm = np.linalg.norm(right, axis=1, keepdims=True)
        
        # 处理平行情况
        mask_parallel = right_norm.flatten() < 1e-6
        if np.any(mask_parallel):
            # 使用不同的上向量
            world_up[mask_parallel] = [0, 1, 0]
            right[mask_parallel] = np.cross(world_up[mask_parallel], forward[mask_parallel])
            right_norm[mask_parallel] = np.linalg.norm(right[mask_parallel], axis=1, keepdims=True)
        
        right = right / (right_norm + 1e-8)
        
        # 4. 计算上向量
        up = np.cross(forward, right)
        up_norm = np.linalg.norm(up, axis=1, keepdims=True)
        up = up / (up_norm + 1e-8)
        
        # 5. 构造旋转矩阵
        # OpenCV坐标系: x右, y下, z前
        rotation_matrices = np.zeros((N, 3, 3))
        rotation_matrices[:, 0, :] = right
        rotation_matrices[:, 1, :] = up  # 注意: OpenCV y轴向下
        rotation_matrices[:, 2, :] = forward
        
        # 转置矩阵 (因为OpenCV使用列向量)
        rotation_matrices = np.transpose(rotation_matrices, (0, 2, 1))
        
        return rotation_matrices

    def get_trajectory(self, radius, direction, num_frames):
        """
        direction: (cam-type, dtheta, dphi, dr, dx, dy)
        """
        traj = direction[1:]
        traj=[float(i) for i in traj]
        mode = direction[0]
        dtheta, dphi, dr, dx, dy = traj
        t = np.linspace(0,1,num_frames)[:,None]
        R = dr * radius
        start_loc = np.array([0,0,0])[None]
        right = np.array([1,0,0])[None]
        fwd = np.array([0,0,1])[None]
        up = np.array([0,-1,0])[None]
        fixed_virtual_target = np.array([dx,dy,radius])[None]
        start_focus_point = np.array([0,0,radius])[None]
        direction_mult = 1 if dphi > 0 else -1
        OUTDOOR_SCALE = 1.0
        if mode == "Rotate":
            
            pose_s, pose_t = self.get_poses(radius, num_frames, traj=traj)
            return pose_s, pose_t

        # ==========================================
        # Part 1: 位置计算
        # ==========================================
        elif mode == "Static":
            current_loc = start_loc

        elif mode == "Orbit":
            # Orbit 本身就是圆滑的单向运动，保留
            orbit_rad = R * 0.95 * OUTDOOR_SCALE
            angle = t * 2.0 * np.pi 
            off_r = orbit_rad * (np.cos(angle) - 1.0)
            off_u = orbit_rad * np.sin(angle)
            current_loc = start_loc + (right * off_r * direction_mult) + (up * off_u)

        elif mode == "Helix_In":
            # 螺旋突进 (这个通常很稳，可以保留)
            push_dist = R * 0.95 * OUTDOOR_SCALE
            spiral_rad = R * 0.4 * OUTDOOR_SCALE
            freq = 2.4 * np.pi 
            vec_push = fwd * push_dist * t
            vec_spiral = (right * (np.cos(t * freq) - 1.0) * spiral_rad * direction_mult) + \
                         (up * np.sin(t * freq) * spiral_rad)
            current_loc = start_loc + vec_push + vec_spiral

        elif mode == "S_Curve_Dolly":
            # S型 (如果这个没抖动就保留，有抖动也可以换)
            side_dist = R * 1.3 * OUTDOOR_SCALE
            depth_amp = R * 0.6 * OUTDOOR_SCALE
            vec_side = right * side_dist * t * direction_mult
            vec_depth = fwd * np.sin(t * 2 * np.pi) * depth_amp
            current_loc = start_loc + vec_side + vec_depth

        elif mode == "Zoom-out":
            current_loc = start_loc - (fwd * R * t)

        elif mode == "Orbit_Ascend":
            # 【新动作 2：盘旋升空】 (替代 Crane_Lift_Away)
            # 动作：Orbit (左右摇摆) + 向上升 (Up)
            # 优势：模拟无人机绕塔上升
            
            # 1. 左右大幅度摇摆 (半个Orbit)
            swing_rad = R * 0.9 * OUTDOOR_SCALE
            # sin(t * pi) -> 0 -> 1 -> 0 (画个拱门)
            vec_swing = right * np.sin(t * np.pi) * swing_rad * direction_mult
            
            # 2. 前后微调 (保持距离感)
            vec_depth = fwd * np.sin(t * 2 * np.pi) * (R * 0.2)
            
            # 3. 大幅度上升
            vec_up = up * (R * 0.9 * OUTDOOR_SCALE) * t
            
            current_loc = start_loc + vec_swing + vec_depth + vec_up

        elif mode == "Parabola_Dive":
            # U型 (保留)
            slide_dist = R * 0.9 * OUTDOOR_SCALE
            drop_height = R * 0.9 * OUTDOOR_SCALE
            vec_slide = right * slide_dist * t * direction_mult
            vec_height = up * (np.cos(t * 2 * np.pi) - 1.0) * 0.5 * drop_height
            vec_push = fwd * np.sin(t * np.pi) * (R * 0.4)
            current_loc = start_loc + vec_slide + vec_height + vec_push
        else:
            raise Exception("Invalid Trajectory Mode:", mode)
        
        lerp_vec = (fixed_virtual_target - start_focus_point) * t
        current_focus_point = start_focus_point + lerp_vec
        # N,3,3
        # target_rot = unreal.MathLibrary.find_look_at_rotation(current_loc, current_focus_point)
        target_rot = self.find_look_at_rotation(current_loc, current_focus_point, up)
        c2ws = np.eye(4)[None].repeat(num_frames, 0)
        c2ws[:,:3,:3] = target_rot
        c2ws[:,:3,3] = current_loc
        pose_t = np.linalg.inv(c2ws)
        pose_s = pose_t[0:1].repeat(num_frames, 0)
        return torch.asarray(pose_s).float(), torch.asarray(pose_t).float()