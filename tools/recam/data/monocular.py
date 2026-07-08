
import torch
import sys, os
import json
import numpy as np
import cv2
from PIL import Image
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime
import random
import gc
from torchvision import transforms
from utils.ioutils import save_video
from utils.geometry import align_inv_depth_to_depth
from utils.geometry import unproject_depth_map_to_point_map
class DA3():
    def __init__(self, save_dir):
        
        # target_path = os.path.join(project_root, "3rdparties", "Depth-Anything-3")
        # print(f"\n[DEBUG] Project Root: {project_root}")
        # print(f"[DEBUG] Trying to append path: {target_path}")
        
        # if os.path.exists(target_path):
        #     print(f"[DEBUG] Path exists. Contents: {os.listdir(target_path)}")
        # else:
        #     print(f"[DEBUG] Path does NOT exist! Check folder name and case sensitivity.")


        sys.path.append(os.path.join(project_root, "3rdparties", "Depth-Anything-3", "src"))

        from depth_anything_3.api import DepthAnything3
        self.device = torch.device("cuda")
        self.model = DepthAnything3.from_pretrained("checkpoints/depth-anything/DA3NESTED-GIANT-LARGE").to("cpu")
        self.model.eval()

        self.save_dir = save_dir

    def video_depth(self, videos=None, video_path = None):
        import glob
        images_dir = f"/tmp/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_{random.randint(0,10000)}"
        print(images_dir)
        try:
            os.makedirs(images_dir)
        except:
            images_dir = f"/tmp/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_{random.randint(0,10000)}"
            print(images_dir)
            os.makedirs(images_dir)
        if video_path is None:
            save_video(
                videos,
                os.path.join(images_dir, f'input.mp4'),
                fps=30
            )
            video_path = os.path.join(images_dir, f'input.mp4')
        os.system((f"ffmpeg -i {video_path} -loglevel quiet -qscale:v 1 -qmin 1 {images_dir}/%04d.png"))
        img = cv2.imread(f"{images_dir}/0001.png")
        original_h, original_w, _ = img.shape
        images = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.model = self.model.to(self.device)
        prediction = self.model.inference(
            images,
            export_format="glb", #(mini_npz, npz, glb, ply, gs, gs_video)
            export_dir=self.save_dir,
        )
        # prediction.processed_images : [N, H, W, 3] uint8   array
        N,H,W,_ = prediction.processed_images.shape
        info = {
            "frames": prediction.processed_images / 255, # N,H,W,3
            "depth": prediction.depth[...,None], # N,H,W,1
            "depth_map_conf": prediction.conf[...,None], # N,H,W,1
            "K":  prediction.intrinsics,
            "pose": np.concatenate([ prediction.extrinsics, np.repeat(np.array([[[0,0,0,1]]]), N, axis=0)], axis=-2),
        }
        # Resize
        depth_conf = torch.asarray(info['depth_map_conf'], device=self.device).permute(0,3,1,2)
        depth_map = torch.asarray(info['depth'], device=self.device).permute(0,3,1,2)
        frames = torch.asarray(info['frames'], device=self.device).permute(0,3,1,2)
        K = info['K']
        print("Resize to ",(original_h, original_w))
        resize_transform = transforms.Resize((original_h, original_w), interpolation=Image.BILINEAR)
        K[:,0,:] *= original_w / depth_map.shape[-1]
        K[:,1,:] *= original_h / depth_map.shape[-2]
        depth_map = resize_transform(depth_map)
        depth_conf = resize_transform(depth_conf)
        frames = resize_transform(frames)
        self.model = self.model.to("cpu")
        
        
        
        info['frames'] = frames.permute(0,2,3,1).cpu().numpy()
        info['depth'] = depth_map.permute(0,2,3,1).cpu().numpy()
        info['depth_map_conf'] = depth_conf.permute(0,2,3,1).cpu().numpy()
        info['K'] = K
        gc.collect()
        
        info['depth'] = info['depth'] * 5
        info['pose'][:,:3,3] = info['pose'][:,:3,3] * 5 
        return info

class Align_DA3_DC(DA3):
    def __init__(self, save_dir, unet_path="checkpoints/tencent/DepthCrafter", pre_train_path="checkpoints/stabilityai/stable-video-diffusion-img2vid"):
        from utils.dc_infer import DepthCrafterDemo
        self.depth_estimater = DepthCrafterDemo(
            unet_path=unet_path,
            pre_train_path=pre_train_path,
            cpu_offload="sequential", # use 'model' will be faster
            device="cuda",
        )
        self.depth_estimater.training = False #FIX
        super().__init__(save_dir)
    
    def video_depth(self, videos):
        """
        videos: L,H,W,3
        """
        depths, ori_depths = self.depth_estimater.infer(
            videos,
            0.0001,
            10000,
            5,
            1.0,
            window_size=110,
            overlap=25,
        )
        gc.collect()
        
        ori_depths = ori_depths.to(self.device)  
        da3_result = super().video_depth(videos)        
        depth_conf = torch.asarray(da3_result['depth_map_conf'], device=self.device).permute(0,3,1,2)
        depth_map = torch.asarray(da3_result['depth'], device=self.device).permute(0,3,1,2)
        K = da3_result['K']
        mask = (depth_map.to(self.device) < (depth_map.mean()+depth_map.min())/2)
        print("Only use near points to align, mask mean:", mask.float().mean().item())
        if mask.float().mean() > 0.2:
            print("Use da3 to align")
            aligned_depth, scale, bias = align_inv_depth_to_depth(ori_depths.to(self.device)  , depth_map.to(self.device) , mask , quantile_masking=False)
            aligned_depth[aligned_depth < 1e-2] = 1000
            info = {
                "frames": videos,
                "depth": aligned_depth.permute(0,2,3,1).cpu().numpy(),
                "K": K,
                "pose": da3_result['pose'],
            }
            self.scale = scale
            self.shift = bias
        else:
            print("da3 invalid")
            cx = 512.0  # depths.shape[-1]//2
            cy = 288.0  # depths.shape[-2]//2
            f = 500  # 500.
            K = (
                torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
                .repeat(videos.shape[0], 1, 1)
            ).cpu().numpy()

            info = {
                "frames": videos,
                "depth": depths.permute(0,2,3,1).cpu().numpy(),
                "K": K
            }
            self.scale = 2.0
            self.shift = 0
        import trimesh
        i = 0
        p1 = unproject_depth_map_to_point_map(da3_result['depth'][i:i+1], da3_result['pose'][i:i+1], da3_result['K'][i:i+1])
        trimesh.PointCloud(p1[0,::10,::10].reshape(-1,3)).export(f"{self.save_dir}/da3.ply")
        p2 = unproject_depth_map_to_point_map(info['depth'][i:i+1], da3_result['pose'][i:i+1], da3_result['K'][i:i+1])
        trimesh.PointCloud(p2[0,::10,::10].reshape(-1,3)).export(f"{self.save_dir}/dc.ply")
        np.save(f"{self.save_dir}/da3_depth.npy", da3_result['depth'][:1])
        np.save(f"{self.save_dir}/dc_depth.npy", depths[:1])
        return info

class DepthCrafter():
    def __init__(self, save_dir, unet_path="checkpoints/tencent/DepthCrafter", pre_train_path="checkpoints/stabilityai/stable-video-diffusion-img2vid"):
        from utils.dc_infer import DepthCrafterDemo
        self.depth_estimater = DepthCrafterDemo(
            unet_path=unet_path,
            pre_train_path=pre_train_path,
            cpu_offload="model",
            device="cuda",
        )
        self.depth_estimater.training = False #FIX
        self.device = torch.device("cuda")
    
    def video_depth(self, videos):
        """
        videos: L,H,W,3
        """
        num_frames = videos.shape[0]
        depths, ori_depths = self.depth_estimater.infer(
            videos,
            0.0001,
            10000,
            5,
            1.0,
            window_size=110,
            overlap=25,
        )

        cx = 512.0  # depths.shape[-1]//2
        cy = 288.0  # depths.shape[-2]//2
        f = 500  # 500.
        K = (
            torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            .repeat(num_frames, 1, 1)
        ).cpu().numpy()

        return {
            "frames": videos,
            "depth": depths.permute(0,2,3,1).cpu().numpy(),
            "K": K
        }

class GeometryCrafter():
    def __init__(self, save_dir=None):
        from utils.geo_infer import GeometryCrafterDemo
        
        self.device = torch.device("cuda")
        self.depth_estimater = GeometryCrafterDemo(
            device=self.device
        )
    
    def video_depth(self, videos):
        from utils.geometry_torch import recover_focal_shift
        
        frames = videos
        points_map, mask = self.depth_estimater.infer(
            frames,
            5,
            1.0,
            window_size=110,
            overlap=25,
        )  
        t,h,w,c = points_map.shape
        # trimesh.PointCloud(points_map[-1].reshape([-1,3]).clip(-10,10).cpu().numpy(), (frames[-1].permute(1,2,0).cpu().numpy().reshape([-1,3])*255).astype(np.uint8)).export(f"{self.save_dir}/geocrafter.ply")
        depths = points_map[...,-1:]
        while depths.min() < 2:
            depths = depths * 2
        focal, shift = recover_focal_shift(points_map, mask) # use recover_focal_shift function from MoGe
        fx = fy = focal * (h ** 2 + w ** 2) ** 0.5
        cx = w / 2
        cy = h / 2
        # intr = utils3d.intrinsics_from_focal_center(fx, fy, cx, cy)
        zeros, ones = torch.zeros_like(fx), torch.ones_like(fx)
        intr = torch.stack([
            fx, zeros, torch.full_like(fx, cx), 
            zeros, fy, torch.full_like(fx, cy), 
            zeros, zeros, ones
        ], dim=-1).unflatten(-1, (3, 3))
        info = {
            "frames": frames,
            "depth": depths.cpu().numpy(),
            "K": intr.cpu().numpy()
        }
        return info

class Pi3():
    
    def __init__(self, save_dir=None):
        sys.path.append(os.path.join(project_root, "3rdparties","pi3"))
        from pi3.models.pi3 import Pi3
        self.device = torch.device("cuda")
        self.depth_model = Pi3.from_pretrained("checkpoints/yyfz233/Pi3").to(self.device).eval()
        

    @torch.no_grad()
    def video_depth(self, videos):
        from utils.geometry_torch import recover_focal_shift
        
        # all frames calculate depth
        videos = videos.clip(0,1)
        videos_ch = torch.asarray(videos).permute(0,3,1,2)
        # Resize video
        t, c, h, w = videos_ch.shape
        new_width = 504 #FIX?
        new_height = round(h * (new_width / w) / 14) * 14
        resize_transform = transforms.Resize((new_height, new_width), interpolation=Image.BILINEAR)
        video_resized = resize_transform(videos_ch)  # [T, C, H, W]
        if new_height > 504:
            start_y = (new_height - 504) // 2
            video_resized = video_resized[:, :, start_y:start_y + 504, :]
            resize_depths = resize_depths[:, start_y:start_y + 504, :]
        video_resized = video_resized.clip(0, 1).to(self.device) #FIX
        
        # Run inference
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                res = self.depth_model(video_resized[None]) # Add batch dimension
        
        frames = video_resized                        # N x C x H x W
        points = res["points"].squeeze(0)             # N x H x W x 3
        local_points_conf = res["conf"].squeeze(0)    # N x H x W x 1
        local_points = res["local_points"].squeeze(0) # N x H x W x 3
        poses = torch.inverse(res["camera_poses"].squeeze(0))        # N x 4 x 4
        
        # Use res["local_points"] to recover intrinsic parameters
        masks_recover = torch.sigmoid(res["conf"][..., 0]) > 0.1
        original_height, original_width = res["local_points"].shape[-3:-1]
        aspect_ratio = original_width / original_height
        
        focal, shift = recover_focal_shift(local_points, masks_recover) # use recover_focal_shift function from MoGe
        fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
        
        fx *= new_width
        fy *= new_height
        cx, cy = new_width / 2, new_height / 2
        # intr = utils3d.intrinsics_from_focal_center(fx, fy, cx, cy)
        zeros, ones = torch.zeros_like(fx), torch.ones_like(fx)
        intr = torch.stack([
            fx, zeros, torch.full_like(fx, cx), 
            zeros, fy, torch.full_like(fx, cy), 
            zeros, zeros, ones
        ], dim=-1).unflatten(-1, (3, 3))
        
        # Inverse resize transform
        inv_resize_transform = transforms.Resize((h, w), interpolation=Image.BILINEAR)
        frames = inv_resize_transform(frames)                                                    # N x C x H x W
        depth_map = inv_resize_transform(local_points[:, :, :, 2:3].permute(0, 3, 1, 2))         # N x 1 x H x W
        depth_map_conf = inv_resize_transform(local_points_conf.permute(0, 3, 1, 2))             # N x H x W

        intr[:,:2,:2] = intr[:,:2,:2] * w / new_width
        intr[:,0,2] = w / 2
        intr[:,1,2] = h / 2

        # Record and save results
        info = {
            "frames": frames.permute(0,2,3,1).cpu().numpy(),
            "depth": depth_map.permute(0,2,3,1).cpu().numpy(),
            "depth_map_conf": depth_map_conf.permute(0,2,3,1).cpu().numpy(),
            "pose": poses.cpu().numpy(),
            "K": intr.cpu().numpy(),
        }
        info['depth'] = info['depth'] * 5
        info['pose'][:,:3,3] = info['pose'][:,:3,3] * 5 
        return info



class Pi3X():
    
    def __init__(self, save_dir=None):
        sys.path.append(os.path.join(project_root, "3rdparties","pi3"))
        # from pi3.models.pi3 import Pi3X
        from pi3.models.pi3x import Pi3X
        self.device = torch.device("cuda")
        self.depth_model = Pi3X.from_pretrained("checkpoints/yyfz233/Pi3X").to(self.device).eval()
        

    @torch.no_grad()
    def video_depth(self, videos, conditions={}):
        from utils.geometry_torch import recover_focal_shift
        # all frames calculate depth
        videos = videos.clip(0,1)
        videos_ch = torch.asarray(videos).permute(0,3,1,2)
        # Resize video
        t, c, h, w = videos_ch.shape
        new_width = 504 #FIX?
        new_height = round(h * (new_width / w) / 14) * 14
        resize_transform = transforms.Resize((new_height, new_width), interpolation=Image.BILINEAR)
        video_resized = resize_transform(videos_ch)  # [T, C, H, W]
        if new_height > 504:
            start_y = (new_height - 504) // 2
            video_resized = video_resized[:, :, start_y:start_y + 504, :]
            resize_depths = resize_depths[:, start_y:start_y + 504, :]
        video_resized = video_resized.clip(0, 1).to(self.device) #FIX
        
        # Run inference
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        for key in conditions.keys():
            conditions[key] = torch.asarray(conditions[key][None]).to(device=self.device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                res = self.depth_model(video_resized[None], **conditions) # Add batch dimension
        
        frames = video_resized                        # N x C x H x W
        points = res["points"].squeeze(0)             # N x H x W x 3
        local_points_conf = res["conf"].squeeze(0)    # N x H x W x 1
        local_points = res["local_points"].squeeze(0) # N x H x W x 3
        poses = torch.inverse(res["camera_poses"].squeeze(0))        # N x 4 x 4

        # Use res["local_points"] to recover intrinsic parameters
        masks_recover = torch.sigmoid(res["conf"][..., 0]) > 0.1
        original_height, original_width = res["local_points"].shape[-3:-1]
        aspect_ratio = original_width / original_height
        
        focal, shift = recover_focal_shift(local_points, masks_recover) # use recover_focal_shift function from MoGe
        fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
        
        fx *= new_width
        fy *= new_height
        cx, cy = new_width / 2, new_height / 2
        # intr = utils3d.intrinsics_from_focal_center(fx, fy, cx, cy)
        zeros, ones = torch.zeros_like(fx), torch.ones_like(fx)
        intr = torch.stack([
            fx, zeros, torch.full_like(fx, cx), 
            zeros, fy, torch.full_like(fx, cy), 
            zeros, zeros, ones
        ], dim=-1).unflatten(-1, (3, 3))
        # Inverse resize transform
        inv_resize_transform = transforms.Resize((h, w), interpolation=Image.BILINEAR)
        frames = inv_resize_transform(frames)                                                    # N x C x H x W
        depth_map = inv_resize_transform(local_points[:, :, :, 2:3].permute(0, 3, 1, 2))         # N x 1 x H x W
        depth_map_conf = inv_resize_transform(local_points_conf.permute(0, 3, 1, 2))                                             # N x H x W

        intr[:,:2,:2] = intr[:,:2,:2] * w / new_width
        intr[:,0,2] = w / 2
        intr[:,1,2] = h / 2

        # Record and save results
        info = {
            "frames": frames.permute(0,2,3,1).cpu().numpy(),
            "depth": depth_map.permute(0,2,3,1).cpu().numpy(),
            "depth_map_conf": depth_map_conf.permute(0,2,3,1).cpu().numpy(),
            "pose": poses.cpu().numpy(),
            "K": intr.cpu().numpy(),
        }
        
        info['depth'] = info['depth'] * 5
        info['pose'][:,:3,3] = info['pose'][:,:3,3] * 5 
        return info