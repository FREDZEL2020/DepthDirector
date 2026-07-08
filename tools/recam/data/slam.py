import torch
import sys, os
import json
import numpy as np
import cv2
from PIL import Image
import open3d as o3d
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime
import random
import gc
from torchvision import transforms
from utils.ioutils import save_video
from pathlib import Path
import OpenEXR
class Vipe():
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def read_depth_artifacts(self, zip_file_path: Path):
        """
        Read metric depth from zipped exr files.
        """
        import zipfile
        valid_width, valid_height = 0, 0
        with zipfile.ZipFile(zip_file_path, "r") as z:
            for file_name in sorted(z.namelist()):
                frame_idx = int(file_name.split(".")[0])
                with z.open(file_name) as f:
                    try:
                        exr = OpenEXR.InputFile(f)
                    except OSError:
                        # Sometimes EXR loader might fail, we return all nan maps.
                        print(f"Failed to load EXR file {zip_file_path}-{file_name}. Returning all nan maps.")
                        assert valid_width > 0 and valid_height > 0
                        yield (
                            frame_idx,
                            torch.full(
                                (valid_height, valid_width),
                                float("nan"),
                                dtype=torch.float32,
                            ),
                        )
                        continue
                    header = exr.header()
                    dw = header["dataWindow"]
                    valid_width = width = dw.max.x - dw.min.x + 1
                    valid_height = height = dw.max.y - dw.min.y + 1
                    channels = exr.channels(["Z"])
                    depth_data = np.frombuffer(channels[0], dtype=np.float16).reshape((height, width, 1))
                    yield depth_data.copy()

    def read_rgb_artifacts(self, rgb_file_path: Path):
        """
        Read RGB from H264-encoded video.
        """
        import imageio
        reader = imageio.get_reader(rgb_file_path, "ffmpeg")
        for frame_idx, rgb in enumerate(reader):
            rgb = rgb / 255.0
            yield rgb

    def video_depth(self, videos=None, video_path=None):
        print(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        if video_path is None:
            save_video(videos, os.path.join(self.save_dir, "input.mp4"))
            video_path = os.path.join(self.save_dir, "input.mp4")
        os.system(f"vipe infer {video_path} --output {self.save_dir}") 
        # inds, data
        c2ws = np.load(f"{self.save_dir}/pose/{os.path.basename(video_path).replace('mp4','npz')}")['data']
        # fx, fy, cx, cy
        intrinsics = np.load(f"{self.save_dir}/intrinsics/{os.path.basename(video_path).replace('mp4','npz')}")['data']
        depths = self.read_depth_artifacts(f"{self.save_dir}/depth/{os.path.basename(video_path).replace('mp4','zip')}")
        # Tensor l,h,w,c
        videos = self.read_rgb_artifacts(f"{self.save_dir}/rgb/{os.path.basename(video_path)}")
        
        depths = np.stack(list(depths))
        videos = np.stack(list(videos))
        K = np.stack([intrinsics[:,0], np.zeros_like(intrinsics[:,0]), intrinsics[:,2], 
                       np.zeros_like(intrinsics[:,0]), intrinsics[:,1], intrinsics[:,3],
                        np.zeros_like(intrinsics[:,0]), np.zeros_like(intrinsics[:,0]), np.ones_like(intrinsics[:,0]) ], axis=-1).reshape([-1,3,3])
        
        info = {
            "frames": videos, # N,H,W,3
            "depth": depths, # N,H,W,1
            "K":  K,
            "pose": np.linalg.inv(c2ws)
        }
        gc.collect()
        os.system(f"rm -r {self.save_dir}")
        return info

class MegaSAM():
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def video_depth(self, videos=None, video_path=None):
        os.makedirs(self.save_dir, exist_ok=True)
        if video_path is None:
            save_video(videos, os.path.join(self.save_dir, "input.mp4"))
            video_path = os.path.join(self.save_dir, "input.mp4")
        code_dir = f"{project_root}/3rdparties/mega-sam"
        os.environ['PWD'] = os.path.abspath(code_dir)
        os.system((f"ffmpeg -i {video_path} -loglevel quiet -qscale:v 1 -qmin 1 {self.save_dir}/%04d.jpg"))
        os.system(f"python {code_dir}/Depth-Anything/run_videos.py --localhub --encoder vitl --load-from {code_dir}/Depth-Anything/checkpoints/depth_anything_vitl14.pth --img-path {self.save_dir} --outdir {self.save_dir}/Depth-Anything/")
        os.system(f"python {code_dir}/UniDepth/scripts/demo_mega-sam.py --scene-name '' --img-path {self.save_dir} --outdir {self.save_dir}/UniDepth/")
        os.system(f"cd {code_dir} && python camera_tracking_scripts/test_demo.py --output_path={self.save_dir} --datapath={self.save_dir} --weights=checkpoints/megasam_final.pth --scene_name '' --mono_depth_path {self.save_dir}/Depth-Anything/ --metric_depth_path {self.save_dir}/UniDepth/ --disable_vis $@")
        results = np.load(f"{self.save_dir}/droid.npz")        
        info = {
            "frames": results['images'], # N,H,W,3
            "depth": results['depths'][...,None], # N,H,W,1
            "K":  np.repeat(results['intrinsic'][None], results['images'].shape[0], axis=0),
            "pose": np.linalg.inv(results['cam_c2w'])
        }
        gc.collect()
        os.system(f"rm -r {self.save_dir}")
        return info