import os
import numpy as np
import cv2
from processor.base import Warper
import json
class UE_Viser(Warper):
    def __init__(self, opts):
        Warper.__init__(self, opts)
        # self.rand_scale = random.randint(1,5)
        
    def process(self):
        depth_path = f"{self.opts.video_path}/depth.npy"
        cam_path = f"{self.opts.video_path}/cameras.json"
        video_path = f"{self.opts.video_path}/target.mp4"
        depth_downsample = np.load(depth_path)[::4]
        df,dh,dw,_ = depth_downsample.shape
        with open(cam_path, "r") as f:
            cam_meta = json.load(f)
        c2ws = cam_meta['c2ws']
        K = np.array([
            [ cam_meta['fl_x'] / 2 / cam_meta['cx'] * dw, 0, dw/2 ],
            [ 0, cam_meta['fl_y'] / 2 / cam_meta['cy'] * dh, dh/2 ],
            [ 0,0,1 ]
        ])
        depth_downsample = depth_downsample
        video = self.load_mono_video(video_path, stride=16)
        video = video[:, ::8, ::8]
        c2ws = np.array(c2ws)[::4]
        predictions = {
            "depth": depth_downsample,
            "depth_conf": (depth_downsample < 100).astype(np.int32) * 10,
            "images": video,
            "extrinsic": np.linalg.inv(c2ws)[:,:3],
            "intrinsic": np.repeat(K[None], df, axis=0)
        }
        self.visualize(predictions, f"{self.opts.video_path}/vis.glb")
        print("Output to ", f"{self.opts.video_path}/vis.glb")