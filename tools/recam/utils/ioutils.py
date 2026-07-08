from decord import VideoReader, cpu
import torch
import numpy as np
import cv2
import torchvision
import os
from PIL import Image

def read_video_frames(video_path, process_length, stride, max_res, dataset="open", start=0):
    if dataset == "depthcrafter":
        print("==> processing video: ", video_path)
        vid = VideoReader(video_path, ctx=cpu(0))
        print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        # FIXME: hard coded
        width = 1024
        height = 576

        target_aspect_ratio = width / height

        # Calculate crop coordinates to maintain aspect ratio
        current_aspect_ratio = original_width / original_height
        if current_aspect_ratio > target_aspect_ratio:
            # Crop width (video is wider than target)
            crop_width = int(original_height * target_aspect_ratio)
            crop_height = original_height
            x = (original_width - crop_width) // 2
            y = 0
        else:
            # Crop height (video is taller than target)
            crop_height = int(original_width / target_aspect_ratio)
            crop_width = original_width
            x = 0
            y = (original_height - crop_height) // 2

        # Create final VideoReader with center crop and resize
        vid = VideoReader(
            video_path,
            ctx=cpu(0),
            width=original_width,
            height=original_height,
        )
    elif dataset == "square":
        print("==> processing video: ", video_path)
        vid = VideoReader(video_path, ctx=cpu(0))
        print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        # FIXME: hard coded
        width = 504
        height = 504

        target_aspect_ratio = width / height

        # Calculate crop coordinates to maintain aspect ratio
        current_aspect_ratio = original_width / original_height
        if current_aspect_ratio > target_aspect_ratio:
            # Crop width (video is wider than target)
            crop_width = int(original_height * target_aspect_ratio)
            crop_height = original_height
            x = (original_width - crop_width) // 2
            y = 0
        else:
            # Crop height (video is taller than target)
            crop_height = int(original_width / target_aspect_ratio)
            crop_width = original_width
            x = 0
            y = (original_height - crop_height) // 2

        # Create final VideoReader with center crop and resize
        vid = VideoReader(
            video_path,
            ctx=cpu(0),
            width=original_width,
            height=original_height,
        )
    elif dataset == "original":
        print("==> processing video: ", video_path)
        vid = VideoReader(video_path, ctx=cpu(0))
        print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        
        # Create final VideoReader with center crop and resize
        vid = VideoReader(
            video_path,
            ctx=cpu(0),
            width=original_width,
            height=original_height,
        )
    else:
        print("==> processing video: ", video_path)
        vid = VideoReader(video_path, ctx=cpu(0))
        print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        
        # FIXME: hard coded
        height = original_height // max_res
        width = original_width // max_res
        aspect_ratio = original_width / original_height
        if aspect_ratio < 1:
            height = max_res
            width = int(max_res * aspect_ratio)
        else:
            width = max_res
            height = int(max_res / aspect_ratio)
        vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
        
    frames_idx = list(range(start, len(vid), stride))
    print(
        f"==> downsampled shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}, with stride: {stride}"
    )
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    print(
        f"==> final processing shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}"
    )
    
    if dataset != "original":
        frames = []
        for i in frames_idx:
            frame = vid[i].asnumpy().astype("float32") / 255.0
            # Apply center crop to original dimensions
            cropped_frame = frame[y:y+crop_height, x:x+crop_width]
            # Resize to target dimensions
            resized_frame = cv2.resize(cropped_frame, (width, height))
            frames.append(resized_frame)
        frames = np.stack(frames)
        print(frames.shape)
    else:
        frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0
    return frames


def save_video(data, images_path, folder=None, fps=8):
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    elif isinstance(data, list):
        folder = [folder] * len(data)
        images = [
            np.array(Image.open(os.path.join(folder_name, path)))
            for folder_name, path in zip(folder, data)
        ]
        stacked_images = np.stack(images, axis=0)
        tensor_data = torch.from_numpy(stacked_images).to(torch.uint8)
    torchvision.io.write_video(
        images_path, tensor_data, fps=fps, video_codec='h264', options={'crf': '10'}
    )