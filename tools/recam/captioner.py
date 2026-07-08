import torch
import os
from tqdm import tqdm
import argparse
import csv
import random
from PIL import Image
from utils.ioutils import read_video_frames
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BlipProcessor
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class ImageCaptionDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        """
        image_paths: 图像路径列表
        transform: 图像预处理变换
        """
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # 加载图像
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            
            # 应用变换（如果有）
            if self.transform:
                image = self.transform(image)
                
            return image, image_path
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个空白图像作为占位符
            blank_image = Image.new('RGB', (224, 224), color='white')
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, image_path

def collate_fn(batch):
    """自定义批次整理函数"""
    images, paths = zip(*batch)
    return list(images), list(paths)

class Captioner():
    def __init__(self, opts):
        self.opts = opts
        self.caption_processor = None
        self.captioner = None

    def _load_model(self):
        if self.captioner is None:
            print(f"Loading captioner model from {self.opts.blip_path}...")
            self.caption_processor = AutoProcessor.from_pretrained(self.opts.blip_path, use_fast=False)
            self.captioner = Blip2ForConditionalGeneration.from_pretrained(
                self.opts.blip_path, torch_dtype=torch.float16
            ).to(self.opts.device)

    def caption(self, frames_list, repl_list):
        opts = self.opts

        # all camera
        prompt_list = []
        for frames, repl in zip(frames_list, repl_list):
            prompt = self.get_caption(opts, frames.permute(0, 2, 3, 1)[opts.video_length // 2].cpu().numpy())
            prompt_list.append(prompt)
        del self.captioner
        del self.caption_processor
        self.captioner = None
        self.caption_processor = None
        return prompt_list

    def get_caption(self, opts, image):
        self._load_model()
        image_array = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        inputs = self.caption_processor(images=pil_image, return_tensors="pt").to(
            opts.device, torch.float16
        )
        generated_ids = self.captioner.generate(**inputs)
        generated_text = self.caption_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return generated_text + opts.refine_prompt

    def get_caption_batch(self, opts, pil_images):
        """
        批量处理多张图像，生成描述
        pil_images: 图像列表，每个元素为PIL.Image对象
        """
        self._load_model()
        inputs = self.caption_processor(
            images=pil_images,
            return_tensors="pt",
            padding=True
        ).to(opts.device, torch.float16)

        generated_ids = self.captioner.generate(**inputs)

        generated_texts = self.caption_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        return [text.strip() + opts.refine_prompt for text in generated_texts]

    def get_caption_image(self, opts, pil_image):
        self._load_model()
        inputs = self.caption_processor(images=pil_image, return_tensors="pt").to(
            opts.device, torch.float16
        )
        generated_ids = self.captioner.generate(**inputs)
        generated_text = self.caption_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return generated_text + opts.refine_prompt

    def process_image_batch(self, opts, image_paths, batch_size=32):
        """
        批量处理图像路径列表
        image_paths: 图像路径列表
        batch_size: 批次大小
        """
        # 创建数据集
        dataset = ImageCaptionDataset(image_paths)
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,  # 保持顺序
            num_workers=2,  # 并行加载 workers
            collate_fn=collate_fn,
            pin_memory=True if opts.device == 'cuda' else False
        )
        
        captions = []
        
        # 批量处理
        for batch_images, batch_paths in tqdm(dataloader):
            # 生成描述
            batch_captions = self.get_caption_batch(opts, batch_images)
            
            # 保存结果
            for path, caption in zip(batch_paths, batch_captions):
                captions.append(caption)
            
            # 释放内存（可选）
            if opts.device == 'cuda':
                torch.cuda.empty_cache()

        return captions

    def process_local(self):
        root_dir = self.opts.root_path
        p_list = []
        v_list = []
        input_list = []
        for root, dirs, files in os.walk(root_dir):
            if "input.png" in files:
                video_path = os.path.join(root, "target.mp4")
                input_path = os.path.join(root, "input.png")
            else:
                continue
            if not os.path.exists(video_path):
                assert False
            video_path = os.path.relpath(video_path, root_dir)
            v_list.append(video_path)
            input_list.append(input_path)

        if self.opts.skip_caption:
            p_list = [self.opts.default_prompt + self.opts.refine_prompt] * len(input_list)
        else:
            p_list = self.process_image_batch(self.opts, input_list)

        for prompt, path in zip(p_list, input_list):
            with open(os.path.join(os.path.dirname(path), "prompts.txt"), "w") as f:
                f.write(prompt)
    
    
    def write_csv(self, root_dir, input_list, p_list):
        sorted_indices = [i for i, _ in sorted(enumerate(input_list), key=lambda x: x[1])]
        
        if self.opts.dataset == "val":
            random.shuffle(sorted_indices)
            sorted_indices = sorted_indices[:8]
        elif self.opts.dataset == "test":
            sorted_indices = sorted_indices[:100]
        elif self.opts.dataset == "test_all":
            sorted_indices = sorted_indices
        elif self.opts.dataset == "overfit":
            random.shuffle(sorted_indices)
            sorted_indices = sorted_indices[:100]
        elif self.opts.dataset == "train":
            random.shuffle(sorted_indices)
            
        print(sorted_indices)
        # 对原列表排序（可选）
        # v_list = [v_list[i] for i in sorted_indices]
        input_list = [input_list[i] for i in sorted_indices]
        p_list = [p_list[i] for i in sorted_indices]

        
        if self.opts.dataset in ["val", "train"] :
            with open(f"{root_dir}/metadata_{self.opts.dataset}.csv", 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['input_image', 'input', 'video', 'cond', 'mask', 'depth', 'prompt'])
                writer.writeheader()
                for input_path, prompt in zip(input_list, p_list):
                    input_path = os.path.relpath(input_path, root_dir)
                    cam_names = []
                    scene_path = os.path.dirname(os.path.dirname(input_path))
                    current_cam_name = os.path.dirname(input_path).split("/")[-1]
                    for cam in os.listdir(f"{root_dir}/{scene_path}"):
                        if cam.endswith(".txt"):
                            continue
                        if cam == current_cam_name:
                            continue
                        cam_names.append(cam)
                    input_name = "reference"
                    writer.writerow({
                        'input_image': input_path, 
                        'video': os.path.join(os.path.dirname(input_path),"target.mp4"), 
                        'input': os.path.join(os.path.dirname(os.path.dirname(input_path)), input_name, "target.mp4"),
                        'cond': os.path.join(os.path.dirname(input_path),"dw_warp.mp4"),
                        'mask': os.path.join(os.path.dirname(input_path),"dw_mask.mp4"),
                        'depth': os.path.join(os.path.dirname(input_path),"dw_depth.mp4"),
                        'prompt': prompt})
        else:
            with open(f"{root_dir}/metadata_{self.opts.dataset}.csv", 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['input_image', 'output', 'input', 'cond', 'mask', 'depth', 'prompt'])
                writer.writeheader()
                for input_path, prompt in zip(input_list, p_list):
                    input_path = os.path.relpath(input_path, root_dir)
                    writer.writerow({
                        'input_image': input_path, 
                        'output': os.path.join(os.path.dirname(input_path),"output.mp4"), 
                        'input': os.path.join(os.path.dirname(input_path),"input.mp4"), 
                        'cond': os.path.join(os.path.dirname(input_path),"dw_warp.mp4"),
                        'mask': os.path.join(os.path.dirname(input_path),"dw_mask.mp4"),
                        'depth': os.path.join(os.path.dirname(input_path),"dw_depth.mp4"),
                        'prompt': prompt})
        print(f"{root_dir}/metadata_{self.opts.dataset}.csv")
        
    def collect(self):
        root_dir = self.opts.root_path
        p_list = []
        v_list = []
        input_list = []
        for root, dirs, files in os.walk(root_dir):
            # 检查当前目录下是否存在 videos.txt
            if "input.mp4" in files:
                video_path = os.path.join(root, "input.mp4")
                input_path = os.path.join(root, "input.png")
            else:
                continue
            if not os.path.exists(video_path):
                assert False
            video_path = os.path.relpath(video_path, root_dir)
            v_list.append(video_path)
            input_list.append(input_path)

        if self.opts.skip_caption:
            p_list = [self.opts.default_prompt + self.opts.refine_prompt] * len(input_list)
        else:
            p_list = self.process_image_batch(self.opts, input_list)

        self.write_csv(root_dir, input_list, p_list)
        
    def process(self):
        root_dir = self.opts.root_path
        p_list = []
        v_list = []
        input_list = []
        for root, dirs, files in os.walk(root_dir):
            # 检查当前目录下是否存在 videos.txt
            if "input.png" in files:
                input_path = os.path.join(root, "input.png")
            else:
                continue
            if not os.path.exists(input_path):
                assert False
            input_list.append(input_path)
        if self.opts.skip_caption:
            p_list = [self.opts.default_prompt + self.opts.refine_prompt] * len(input_list)
        else:
            p_list = self.process_image_batch(self.opts, input_list)
        self.write_csv(root_dir, input_list, p_list)


def get_parser():
    parser = argparse.ArgumentParser()

    ## general
    parser.add_argument('--root_path', type=str, help='Input path')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='The device to use'
    )
    parser.add_argument(
        '--video_length', type=int, default=81, help='Length of the video frames'
    )
    
    parser.add_argument('--blip_path', type=str, default="checkpoints/Salesforce/blip2-opt-2.7b")
    parser.add_argument(
        '--task',
        type=str,
        default="process",
        help='task name',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default="train",
        help='task name',
    )
    parser.add_argument(
        '--negative_prompt',
        type=str,
        default="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
        help='Negative prompt for video generation',
    )
    parser.add_argument(
        '--refine_prompt',
        type=str,
        default=". The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.",
        help='Prompt for video generation',
    )
    
    parser.add_argument(
        '--skip_caption',
        action='store_true',
        help='Skip captioning and use default_prompt directly',
    )
    parser.add_argument(
        '--default_prompt',
        type=str,
        default='',
        help='Default prompt used when skip_caption is set',
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()  # infer config.py
    opts = parser.parse_args()
    opts.weight_dtype = torch.bfloat16
    
    opts.save_dir = opts.root_path
    os.makedirs(opts.save_dir, exist_ok=True)
    pvd = Captioner(opts)
    getattr(pvd, opts.task)()