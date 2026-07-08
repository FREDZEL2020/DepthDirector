import os
import numpy as np
import cv2
from data.ue import UE
from processor.base import Warper
from geometry.dynamic_mesh import Dynamic3DMesh
from tqdm import tqdm
import random
import imageio

class UE_Dynamic_Stream(Warper, UE):
    def __init__(self, opts):
        Warper.__init__(self, opts)
        UE.__init__(self, opts)
        self.rand_scale = np.clip(random.random() * 3 - 2, 0, 1) + 1.0
        self.geometry = Dynamic3DMesh(opts.device)
    
    def write_video_stream(self, frame, writer):
        # 目标分辨率
        target_h, target_w = self.opts.sample_size[0], self.opts.sample_size[1]  # 替换为你想要的分辨率

        resized_frame = cv2.resize(frame[..., :3], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        writer.append_data(resized_frame)
                    
    
    def render_results_stream(self, results, writers, render_keys=[""]):
        for render_key in results:
            if render_key == "extra":
                continue
            video = results[render_key]
            if render_key in ["depth", "dw_depth"]:
                # depth_min = 2.0
                # depth_max = 100
                depth_min = results["extra"]["depth_min"]
                depth_max = results["extra"]["depth_max"]
                rgb_video = [self.map_depth(frame, depth_min=depth_min, depth_max=depth_max) for frame in video]
                for video in rgb_video:
                    self.write_video_stream(video, writers[render_key])
            elif "warp" in render_key:
                # rgb_video = (video*255).astype(np.uint8)
                # self.write_video(rgb_video, f"{save_dir}/{render_key}.mp4")
                for i,sub_key in enumerate(render_keys):
                    rgb_video = (video[...,i*3:(i+1)*3]*255).astype(np.uint8)
                    if len(sub_key) > 0:
                        sub_key = sub_key+"_"
                    for video in rgb_video:
                        self.write_video_stream(video, writers[render_key])
            elif "mask" in render_key:
                rgb_video = (video*255).astype(np.uint8)
                for video in rgb_video:
                    self.write_video_stream(video, writers[render_key])

    def process(self):
        frame_idxs = list(range(0,self.opts.video_length))
        f = open(f"{self.save_dir}/videos.txt" , "w")
        valid_video = []
        # 构建相机轨迹路径 (例如: scene1-down)
        cam_s = "center" if random.random() > 0.2 else "rand_input"
        cam_list = []
        for cam_name in os.listdir(self.root_dir):
            if cam_name.endswith("txt"):
                continue
            cam_list.append(cam_name)
        if cam_s not in cam_list:
            cam_s = cam_list[0]
        cam_list.remove(cam_s)
        s_cams, K = self.get_cameras_by_idxs(cam_s, frame_idxs)
        s_cams = np.array(s_cams)
        # s_frames,s_depths = self.get_all_by_idxs(cam_s, frame_idxs)

        writers = {} 
        fps = int(self.opts.fps)
        output_types = ['dw_warp', 'dw_mask', 'dw_depth', 'warp', 'mask', 'depth', 'input', 'target']
        for i,cam_t in enumerate(cam_list):
            
            if not os.path.exists(f"{self.root_dir}/{cam_t}/camera_params.json"):
                print(f"{self.root_dir}/{cam_t}/camera_params.json not exists")
                continue

            valid_video.append(cam_t)
            
            t_cams, K = self.get_cameras_by_idxs(cam_t, frame_idxs)
            t_cams = np.array(t_cams)
            pose_s = (np.linalg.inv(s_cams)) #@ np.diag([1,-1,-1,1])
            pose_t = (np.linalg.inv(t_cams)) #@ np.diag([1,-1,-1,1])
            save_dir = os.path.join(self.save_dir, cam_t)
            os.makedirs(save_dir, exist_ok=True)
            self.save_cameras(t_cams, K[0], f"{save_dir}")

            
            for type_name in output_types:
                save_path = os.path.join(save_dir, f"{type_name}.mp4")
                writers[type_name] = imageio.get_writer(
                    save_path, fps=fps, codec='libx264', quality=6, macro_block_size=None
                )
            j = 0
            depth_min = None
            for i in frame_idxs:
                t_frames,t_depths = self.get_all_by_idxs(cam_t, [i])
                s_frames,s_depths = self.get_all_by_idxs(cam_s, [i])
                self.geometry.build(s_frames, s_depths, K[i:i+1], pose_s[i:i+1])
                results = self.geometry.render(pose_t[i:i+1], K[i:i+1], base_idx=j)
                if depth_min is None:
                    depth_min = results["extra"]["depth_min"]
                    depth_max = results["extra"]["depth_max"]
                else:
                    results["extra"]["depth_min"] = depth_min
                    results["extra"]["depth_max"] = depth_max
                self.render_results_stream(results, writers)
                rgb_video = (t_frames*255).astype(np.uint8)
                for video in rgb_video:
                    self.write_video_stream(video, writers['target'])
                rgb_video = (s_frames*255).astype(np.uint8)
                for video in rgb_video:
                    self.write_video_stream(video, writers['input'])
                j += 1
            print(f"[Info] Finalizing video files at {save_dir}...")
            for name in writers:
                writers[name].close()


class UE_Dynamic_Stream_Depth(Warper, UE):
    def __init__(self, opts):
        Warper.__init__(self, opts)
        UE.__init__(self, opts)
        self.rand_scale = np.clip(random.random() * 3 - 2, 0, 1) + 1.0
        self.geometry = Dynamic3DMesh(opts.device)