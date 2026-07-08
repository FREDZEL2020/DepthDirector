
import os
from tqdm import tqdm
import json
import OpenEXR
import numpy as np
import cv2
from PIL import Image
import Imath
import torch
from utils.camutils import location_rotation_to_transform_matrix
# sRGB gamma correction from linear space
def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
import time
import random
import math

class UE():
    def __init__(self, opts):
        self.root_dir = opts.video_path
        print("self.rand_scale", self.rand_scale)
        scene_name = os.path.basename(self.root_dir)
        self.raw_dir = opts.raw_dir

    def auto_adjust_exposure(self, img_linear, target_grey=0.18):
        """
        Automatically compute exposure gain based on image brightness, for linear-space EXR HDR images.
        :param img_linear: RGB matrix in linear space (float32), shape (H, W, 3) or (H, W, 4)
        :param target_grey: target neutral grey brightness, default 0.18 (18% grey standard)
        :return: exposure compensation value (EV), float
        """
        import numpy as np

        # 1. Input validation and preprocessing
        if img_linear is None or img_linear.size == 0:
            raise ValueError("Input image is empty, please provide valid EXR image data")

        # Ensure RGB data, ignore alpha channel
        if img_linear.ndim != 3 or img_linear.shape[-1] < 3:
            raise ValueError("Input image must be a 3D array with at least 3 RGB channels")
        rgb_data = img_linear[..., :3]

        # 2. Compute luminance using standard weights, avoiding extreme values
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        luminance = np.dot(rgb_data.reshape(-1, 3), weights)
        luminance = luminance.reshape(rgb_data.shape[:2])

        # Filter invalid values: EXR may contain negatives or zeros, set minimum threshold
        min_valid = 1e-7
        luminance = np.clip(luminance, min_valid, None)

        # 3. Use log median instead of log mean for robustness against outlier brightness regions
        log_lum = np.log(luminance)
        # Ignore extreme outliers based on percentiles
        lower, upper = np.percentile(log_lum, [10, 90])
        valid_log_lum = log_lum[(log_lum >= lower) & (log_lum <= upper)]

        if valid_log_lum.size == 0:
            # Fall back to median to avoid all-black or all-white images
            log_avg_lum = np.exp(np.median(log_lum))
        else:
            log_avg_lum = np.exp(np.mean(valid_log_lum))

        # 4. Compute exposure scale, clamped to reasonable range to prevent over-adjustment
        scale = target_grey / log_avg_lum

        # 5. Compute and print EV compensation
        ev_offset = np.log2(scale)
        print(f"Auto exposure compensation: {ev_offset:.2f} EV")

        return ev_offset
    
    def get_all_by_idxs(self, cam, frame_idx, style="no"):
        mv_depth = []
        mv_frame = []
        with open(f"{self.root_dir}/{cam}/camera_params.json", 'r') as f:
            meta = json.load(f)
        cam_path = os.path.join(self.raw_dir, meta['sequence_name'])
        
        while True:
            all_rendered = True
            for i in (frame_idx):
                idx = meta['poses'][i]['idx']
                # Build EXR file path
                str_idx = "{:04d}".format(idx)
                file_path = os.path.join(cam_path, f"{meta['sequence_name']}..{str_idx}.exr")
                if not os.path.exists(file_path):
                    all_rendered = False
                    break
            if all_rendered:
                break
            print("Waiting for rendering done for scene: ", self.root_dir)
            time.sleep(random.randint(60,120))
        for i in (frame_idx):
            idx = meta['poses'][i]['idx']
            # Build EXR file path
            str_idx = "{:04d}".format(idx)
            file_path = os.path.join(cam_path, f"{meta['sequence_name']}..{str_idx}.exr")
            exr_file = OpenEXR.InputFile(file_path)

            # Get header info and image resolution
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            # Define pixel type
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
            depth_channel = "FinalImageMovieRenderQueue_WorldDepth.R"
            motion_channel = ["FinalImageMovieRenderQueue_MotionVectors.A", "FinalImageMovieRenderQueue_MotionVectors.B", "FinalImageMovieRenderQueue_MotionVectors.G", "FinalImageMovieRenderQueue_MotionVectors.R"]
            # RGB channel names
            rgb_channels = ['R', 'G', 'B']
            # Process depth channel
            if depth_channel in header['channels']:
                # Read channel and convert to NumPy array
                depth_data = exr_file.channel(depth_channel, pixel_type)
                depth_array = np.frombuffer(depth_data, dtype=np.float32).reshape((height, width))


            if all(c in header['channels'] for c in rgb_channels):
                rgb_data = [np.frombuffer(exr_file.channel(c, pixel_type), dtype=np.float32).reshape((height, width))
                            for c in rgb_channels]
                rgb_stack = np.stack(rgb_data, axis=-1)  # H x W x 3
                srgb = linear_to_srgb(rgb_stack)

                # Convert to uint8 format
                rgb_uint8 = (srgb * 255.0).astype(np.uint8)

            if all(c in header['channels'] for c in motion_channel):
                motion_data = [np.frombuffer(exr_file.channel(c, pixel_type), dtype=np.float32).reshape((height, width))
                            for c in motion_channel]
                motion_stack = np.stack(motion_data, axis=-1)  # H x W x 3

            depth = depth_array / 100

            # Add channel dimension (H, W) -> (C, H, W)
            if depth.ndim == 2:
                depth = depth[...,None]
            mv_depth.append(depth.astype(np.float32))
            frame = (rgb_uint8.astype(np.float32) / 255).clip(0,1)
            mv_frame.append(frame)
            
        return np.stack(mv_frame, axis=0), np.stack(mv_depth, axis=0)
    
    def get_jpg_by_idxs(self, cam, frame_idx):
        mv_depth = []
        mv_frame = []
        with open(f"{self.root_dir}/{cam}/camera_params.json", 'r') as f:
            meta = json.load(f)
        cam_path = os.path.join(self.raw_dir, meta['sequence_name'])
        
        while True:
            all_rendered = True
            for i in (frame_idx):
                idx = meta['poses'][i]['idx']
                # Build file path
                str_idx = "{:04d}".format(idx)
                file_path = os.path.join(cam_path, f"{meta['sequence_name']}.FinalImage.{str_idx}.jpeg")
                if not os.path.exists(file_path):
                    all_rendered = False
                    break
            if all_rendered:
                break
            print("Waiting for rendering done for scene: ", self.root_dir)
            time.sleep(random.randint(60,120))
            
        for i in (frame_idx):
            idx = meta['poses'][i]['idx']
            str_idx = "{:04d}".format(idx)
            file_path = os.path.join(cam_path, f"{meta['sequence_name']}.FinalImage.{str_idx}.jpeg")
            rgb_uint8 = cv2.imread(file_path)[...,::-1]
            frame = (rgb_uint8.astype(np.float32) / 255).clip(0,1)
            mv_frame.append(frame)
            
        return np.stack(mv_frame, axis=0)
    
    def get_jpg_by_idxs_to_mp4(self, cam, frame_idx, output_mp4_path, fps=30, max_wait_time=10):
        """
        Read jpegs in order and stream-write to MP4.

        Args:
            cam: camera identifier
            frame_idx: list of frame indices
            output_mp4_path: output MP4 file path
            fps: video frame rate, default 30
            max_wait_time: maximum wait time (seconds), default 10
        """
        with open(f"{self.root_dir}/{cam}/camera_params.json", 'r') as f:
            meta = json.load(f)
        cam_path = os.path.join(self.raw_dir, meta['sequence_name'])
        
        # Initialize video writer
        video_writer = None
        frame_count = 0
        start_time = time.time()
        reference_frame = None
        if "offset" in meta:
            offset = int(meta["offset"])
        else:
            offset = 0
        print("offset", offset)
        for i in tqdm(frame_idx, desc=f"Processing frames for {cam}"):
            idx = meta['poses'][i]['idx']
            str_idx = "{:04d}".format(idx)
            file_path = os.path.join(cam_path, f"{meta['sequence_name']}.FinalImage.{str_idx}.jpeg")
            
            # Wait for file to appear
            file_found = False
            wait_start = time.time()

            for i in range(720): # wait up to 7200 seconds
                if os.path.exists(file_path):
                    file_found = True
                    break
                time.sleep(10)  # check every 10 seconds
            if not file_found:
                raise(f"Warning: File {file_path} not found after {max_wait_time}s, skipping frame {i}")
                
            
            # Read image
            rgb_uint8 = cv2.imread(file_path)

            # Initialize video writer using dimensions from first frame
            if video_writer is None:
                height, width = rgb_uint8.shape[:2]
                # Use H264 encoding
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    output_mp4_path, 
                    fourcc, 
                    fps, 
                    (width, height)
                )
                if not video_writer.isOpened():
                    print(f"Error: Failed to open video writer for {output_mp4_path}")
                    return None
                first_frame = (rgb_uint8.astype(np.float32) / 255).clip(0, 1)[...,::-1]
            # Write frame to MP4
            video_writer.write(rgb_uint8)
            frame_count += 1
            if reference_frame is None:
                reference_frame = (rgb_uint8.astype(np.float32) / 255).clip(0, 1)[...,::-1]
            
        # Release video writer
        if video_writer is not None:
            video_writer.release()

            # Force reformat with ffmpeg for broad player compatibility
            # H264 with yuv420p has the best compatibility with browsers and VS Code
            import subprocess
            temp_path = output_mp4_path.replace(".mp4", "_temp.mp4")
            os.rename(output_mp4_path, temp_path)

            cmd = [
                'ffmpeg', '-y', '-i', temp_path,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', # key: determines playback in browsers/VS Code
                '-crf', '23',
                output_mp4_path
            ]

            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.remove(temp_path) # remove temp file
                print(f"Video converted for VS Code compatibility.")
            except Exception as e:
                os.rename(temp_path, output_mp4_path) # restore on failure
                print(f"FFmpeg conversion failed: {e}")

            print(f"Video saved to {output_mp4_path}")
        return reference_frame

    def get_depth_by_idxs(self, cam, frame_idx):
        mv_depth = []
        mv_frame = []
        with open(f"{self.root_dir}/{cam}/camera_params.json", 'r') as f:
            meta = json.load(f)
        cam_path = os.path.join(self.raw_dir, meta['sequence_name'])
        
        while True:
            all_rendered = True
            for i in (frame_idx):
                idx = meta['poses'][i]['idx']
                # Build EXR file path
                str_idx = "{:04d}".format(idx)
                file_path = os.path.join(cam_path, f"{meta['sequence_name']}..{str_idx}.exr")
                if not os.path.exists(file_path):
                    all_rendered = False
                    break
            if all_rendered:
                break
            print("Waiting for rendering done for scene: ", self.root_dir)
            time.sleep(random.randint(60,120))

        for i in (frame_idx):
            idx = meta['poses'][i]['idx']
            # Build EXR file path
            str_idx = "{:04d}".format(idx)
            file_path = os.path.join(cam_path, f"{meta['sequence_name']}..{str_idx}.exr")
            exr_file = OpenEXR.InputFile(file_path)

            # Get header info and image resolution
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            # Define pixel type
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
            depth_channel = "FinalImageMovieRenderQueue_WorldDepth.R"
            # Process depth channel
            if depth_channel in header['channels']:
                # Read channel and convert to NumPy array
                depth_data = exr_file.channel(depth_channel, pixel_type)
                depth_array = np.frombuffer(depth_data, dtype=np.float32).reshape((height, width))

            depth = depth_array / 100

            # Add channel dimension (H, W) -> (C, H, W)
            if depth.ndim == 2:
                depth = depth[...,None]
            # depth = torch.from_numpy(depth).float().to(self.device)
            mv_depth.append(depth.astype(np.float32))
            
        return np.stack(mv_depth, axis=0)
    
    def get_motion_by_idxs(self, cam, frame_idx):
        motions = []
        with open(f"{self.root_dir}/{cam}/camera_params.json", 'r') as f:
            meta = json.load(f)
        cam_path = os.path.join(self.raw_dir, meta['sequence_name'])
        
        while True:
            all_rendered = True
            for i in (frame_idx):
                idx = meta['poses'][i]['idx']
                # Build EXR file path
                str_idx = "{:04d}".format(idx)
                file_path = os.path.join(cam_path, f"{meta['sequence_name']}.Motion.{str_idx}.exr")
                if not os.path.exists(file_path):
                    all_rendered = False
                    break
            if all_rendered:
                break
            print("Waiting for rendering done for scene: ", self.root_dir)
            time.sleep(random.randint(60,120))

        for i in (frame_idx):
            idx = meta['poses'][i]['idx']
            # Build EXR file path
            str_idx = "{:04d}".format(idx)
            file_path = os.path.join(cam_path, f"{meta['sequence_name']}.Motion.{str_idx}.exr")
            exr_file = OpenEXR.InputFile(file_path)

            # Get header info and image resolution
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            # Define pixel type
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
            motion_channel = ["FinalImageMovieRenderQueue_MotionVectors.A", "FinalImageMovieRenderQueue_MotionVectors.B", "FinalImageMovieRenderQueue_MotionVectors.G", "FinalImageMovieRenderQueue_MotionVectors.R"]
            
            if all(c in header['channels'] for c in motion_channel):
                motion_data = [np.frombuffer(exr_file.channel(c, pixel_type), dtype=np.float32).reshape((height, width))
                            for c in motion_channel]
                motion_stack = np.stack(motion_data, axis=-1)  # H x W x 3

            
            motions.append(motion_stack.astype(np.float32))
            
        return np.stack(motions, axis=0)
    
    def get_frame_number(self, cam):
        with open(f"{self.root_dir}/{cam}/camera_params.json", 'r') as f:
            meta = json.load(f)
        return len(meta['poses'])

    def get_cameras_by_idxs(self, cam, frame_idx):
        cameras = []
        K = []
        with open(f"{self.root_dir}/{cam}/camera_params.json", 'r') as f:
            meta = json.load(f)
            for i in frame_idx:
                
                camera_data = meta['poses'][i]
                pos = np.array(camera_data["location"]) / 100
                c2w = location_rotation_to_transform_matrix(
                    pos,
                    camera_data["rotation"]
                )
                # w2c = np.linalg.inv(c2w)
                cameras.append(c2w)
                h,w = meta['intrinsics']['height'], meta['intrinsics']['width']
                fov = meta['intrinsics']['hfov_deg'] # 37.49 # 
                fx = w / (2*np.tan(np.deg2rad(fov)/2))
                fy = fx # * (h / w)
                # fx = fy = 1000
                cx,cy = meta['intrinsics']['cx'], meta['intrinsics']['cy']
                intr = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0,0,1]
                ])
                K.append(intr)
        return cameras, K
    
    def get_times_by_idxs(self, cam, frame_idx):
        times = []
        with open(f"{self.root_dir}/{cam}/camera_params.json", 'r') as f:
            meta = json.load(f)
            for i in frame_idx:
                
                time = meta['times'][i]
                times.append(time)
        return np.array(times)
 
    def remove_all_by_idxs(self, cam, frame_idx):
        with open(f"{self.root_dir}/{cam}/camera_params.json", 'r') as f:
            meta = json.load(f)
        cam_path = os.path.join(self.raw_dir, meta['sequence_name'])

        for i in (frame_idx):
            idx = meta['poses'][i]['idx']
            # Build file path
            str_idx = "{:04d}".format(idx)
            file_path = os.path.join(cam_path, f"{meta['sequence_name']}..{str_idx}.exr")
            if os.path.exists(file_path):
                os.system(f"rm {file_path}")

            file_path = os.path.join(cam_path, f"{meta['sequence_name']}.FinalImage.{str_idx}.jpeg")
            if os.path.exists(file_path):
                os.system(f"rm {file_path}")
    
    def get_render_idx(self, cam, frame_idx):
        with open(f"{self.root_dir}/{cam}/camera_params.json", 'r') as f:
            meta = json.load(f)

        idx = meta['poses'][frame_idx]['idx']
        return idx