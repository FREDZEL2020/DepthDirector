import os
import numpy as np
import cv2
from data.ue import UE
from processor.base import Warper
from geometry.dynamic_mesh import Dynamic3DMesh
from tqdm import tqdm
import random
class UE_Dynamic(Warper, UE):
    def __init__(self, opts):
        Warper.__init__(self, opts)
        UE.__init__(self, opts)
        self.geometry = Dynamic3DMesh(opts.device)
        
    def process(self):
        frame_idxs = list(range(0,self.opts.video_length))
        with open(f"{self.save_dir}/videos.txt", "w") as f:
            valid_video = []
            # 构建相机轨迹路径 (例如: scene1-down)
            cam_s = "center" if random.random() > 0.2 else "rand"
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
            s_frames,s_depths = self.get_all_by_idxs(cam_s, frame_idxs)
            self.save_cameras(s_cams, K[0], self.save_dir)
            os.makedirs(f"{self.save_dir}/{cam_s}", exist_ok=True)
            if self.opts.save_depth:
                t_rate,h_rate,w_rate = self.opts.depth_downsample_rate
                np.savez(f"{self.save_dir}/{cam_s}/depth.npz", depth=s_depths[::t_rate,::h_rate,::w_rate].astype(np.float16))

            rgb_video = (s_frames*255).astype(np.uint8)
            self.write_video(rgb_video, f"{self.save_dir}/{cam_s}/target.mp4")
            for i,cam_t in enumerate(cam_list):

                if not os.path.exists(f"{self.root_dir}/{cam_t}/camera_params.json"):
                    print(f"{self.root_dir}/{cam_t}/camera_params.json not exists")
                    continue

                valid_video.append(cam_t)

                t_cams, K = self.get_cameras_by_idxs(cam_t, frame_idxs)
                t_cams = np.array(t_cams)

                t_frames,t_depths = self.get_all_by_idxs(cam_t, frame_idxs)
                pose_s = (np.linalg.inv(s_cams)) #@ np.diag([1,-1,-1,1])
                pose_t = (np.linalg.inv(t_cams)) #@ np.diag([1,-1,-1,1])

                save_dir = os.path.join(self.save_dir, cam_t)
                os.makedirs(save_dir, exist_ok=True)
                self.save_cameras(s_cams, K[0], self.save_dir)
                self.save_cameras(t_cams, K[0], f"{save_dir}")
                self.geometry.build(s_frames, s_depths, K, pose_s)
                results = self.geometry.render(pose_t, K)
                self.render_results(results, save_dir)
                rgb_video = (t_frames*255).astype(np.uint8)
                self.write_video(rgb_video, f"{save_dir}/target.mp4")
                rgb_video = (s_frames*255).astype(np.uint8)
                self.write_video(rgb_video, f"{save_dir}/input.mp4")

                if self.opts.save_depth:
                    t_rate,h_rate,w_rate = self.opts.depth_downsample_rate
                    np.savez(f"{save_dir}/depth.npz", depth=t_depths[::t_rate,::h_rate,::w_rate].astype(np.float16))
