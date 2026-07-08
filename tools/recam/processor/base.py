
# 
import torch
import sys, os
import json
import numpy as np
import cv2
from PIL import Image
os.environ['DECORD_EOF_RETRY_MAX'] = '20480'
project_root = os.path.dirname(os.path.abspath(__file__))
try:
    sys.path.append(os.path.join(project_root, "3rdparties", "vggt"))
    sys.path.append(os.path.join(project_root, "3rdparties"))
    sys.path.append(os.path.join(project_root))
except:
    print("Warning: vggt not found")

from utils.visual_util import predictions_to_glb, add_cams_to_glb
from utils.geometry import unproject_depth_map_to_point_map
import torchvision.transforms as transforms
from datetime import datetime
from tqdm import tqdm
import open3d as o3d
import trimesh
import math
from imageio.v3 import imread, imwrite
import imageio
import matplotlib
import random
from matplotlib.colors import hsv_to_rgb
import csv
from utils.ioutils import read_video_frames, save_video

class Warper:
    def __init__(self, opts):
        self.device = opts.device
        self.rand_scale = 1.0
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Initialize the model and load the pretrained weights.
        # This will automatically download the model weights the first time it's run, which may take a while.
        self.opts = opts
        path = opts.video_path
        self.save_dir = opts.save_dir
        # self.save_dir = os.path.join(opts.save_dir, path.removesuffix("/").split("/")[-1])
        os.makedirs(self.save_dir, exist_ok= True)

    def render_results(self, results, save_dir, render_keys=[""]):
        os.makedirs(save_dir, exist_ok=True)
        for render_key in results:
            if render_key == "extra":
                continue
            video = results[render_key]
            if render_key in ["depth", "dw_depth"]:
                depth_min = results["extra"]["depth_min"]
                depth_max = results["extra"]["depth_max"]
                rgb_video = [self.map_depth(frame, depth_min=depth_min, depth_max=depth_max) for frame in video]
                self.write_video(rgb_video, f"{save_dir}/{render_key}.mp4")
            elif "warp" in render_key:
                # rgb_video = (video*255).astype(np.uint8)
                # self.write_video(rgb_video, f"{save_dir}/{render_key}.mp4")
                for i,sub_key in enumerate(render_keys):
                    rgb_video = (video[...,i*3:(i+1)*3]*255).astype(np.uint8)
                    if len(sub_key) > 0:
                        sub_key = sub_key+"_"
                    self.write_video(rgb_video, f"{save_dir}/{sub_key}{render_key}.mp4")
            elif "mask" in render_key:
                rgb_video = (video*255).astype(np.uint8)
                self.write_video(rgb_video, f"{save_dir}/{render_key}.mp4")

            
        

    def visualize(self, predictions, path):
        print("Computing world points from depth map...")
        depth_map = predictions["depth"]  # (S, H, W, 1)
        world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
        
        predictions["world_points_from_depth"] = world_points
        glbscene, scene_scale = predictions_to_glb(
            predictions,
            conf_thres=3.0,
            filter_by_frames="All",
            mask_black_bg=False,
            mask_white_bg=False,
            show_cam=True,
            mask_sky=False,
            target_dir=None,
            prediction_mode="Pointmap Regression",
            align=('target_cams' not in predictions)
        )
        if 'target_cams' in predictions:
            target_cams = predictions['target_cams']
            glbscene = add_cams_to_glb(
                target_cams,
                glbscene,
                scene_scale
            )
        glbscene.export(file_obj=path)
 
    def load_mono_video(self, video_path, data_type="original", start=0, stride=1):
        frames_num = self.opts.video_length
        frames=read_video_frames(video_path, frames_num, stride, dataset=data_type, max_res=1920, start=start) # FIX:maxres=1280
        frames = frames.clip(0,1)
        return frames

    def save_cameras(self, c2ws, K, output_path):
        pose_info = {
            "fl_x": K[0,0].item(),
            "fl_y": K[1,1].item(),
            "cx": K[0,2].item(),
            "cy": K[1,2].item(),
            # c2w
            "c2ws": c2ws.tolist()
        }
        with open(f"{output_path}/cameras.json", "w") as f:
            json.dump(pose_info, f, indent=1)

    
    def map_depth(self, depth, shift=0, depth_min=1, depth_max=100):
        scale = self.rand_scale
        # min_depth = depth[0][...,:3].min()
        # print("min_depth", min_depth)
        depth_map = depth
        depth_map[depth==0] = 0
        near = depth_min / scale
        # inference
        far = min(depth_max, near*50)
        # far = min(depth_max, far)
        # near = np.exp(np.log(near) - (np.log(far) - np.log(near)) / 7) 
        depth_map = depth_map.clip(near, far)
        depth_map = (depth_map + 0.5)
        depth_map[depth==0] = 0
        log_depth = np.log(depth_map)

        # 归一化到[0,1]范围
        depth_map = (log_depth - np.log(near)) / (np.log(far+0.5) - np.log(near))
        
        # mapped_depth = depth / 1000
        depth_map = (depth_map*255).astype(np.uint8)
        cmap = matplotlib.cm.get_cmap('Spectral_r')
        depth_map = (cmap(depth_map)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        return depth_map
         
    def write_video(self, video, path):
        """
        video: L,H,W,3
        """
        # 目标分辨率
        target_h, target_w = self.opts.sample_size[0], self.opts.sample_size[1]  # 替换为你想要的分辨率

        # resize所有帧
        resized_video = []
        for frame in video:
            resized_frame = cv2.resize(frame[..., :3], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            resized_video.append(resized_frame)

        resized_video = np.array(resized_video)

        # 保存视频
        print("Output to:", path)
        imwrite(path, resized_video, macro_block_size=1, fps=self.opts.fps)
    