
import imageio, os, torch, warnings, torchvision, argparse, json
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import numpy as np

class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
    
    def crop_and_resize_numpy(self, image, target_height, target_width):
        """
        输入: image - numpy array of shape (H, W, 3) with float32 dtype
        输出: 裁剪和调整大小后的 numpy array of shape (target_height, target_width, 3) with float32 dtype
        """
        # 确保输入是 float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # 转换为 torch 张量并调整维度顺序为 (C, H, W)
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        else:
            image_tensor = image.permute(2, 0, 1).unsqueeze(0)  # 如果已经是 torch 张量
        
        H, W = image_tensor.shape[2], image_tensor.shape[3]
        
        # 计算缩放比例
        scale = max(target_width / W, target_height / H)
        
        # 计算新的尺寸
        new_H, new_W = int(round(H * scale)), int(round(W * scale))
        
        # 使用双线性插值调整大小
        resized = torch.nn.functional.interpolate(
            image_tensor, 
            size=(new_H, new_W), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 中心裁剪
        start_h = (new_H - target_height) // 2
        start_w = (new_W - target_width) // 2
        
        # 确保裁剪不会超出边界
        start_h = max(0, start_h)
        start_w = max(0, start_w)
        
        # 计算实际的裁剪尺寸
        crop_h = min(target_height, new_H - start_h)
        crop_w = min(target_width, new_W - start_w)
        
        # 进行裁剪
        cropped = resized[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]
        
        # 如果需要填充（当缩放后的图像小于目标尺寸时）
        if crop_h < target_height or crop_w < target_width:
            pad_h = target_height - crop_h
            pad_w = target_width - crop_w
            
            # 计算填充
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            # 使用边缘填充
            cropped = F.pad(
                cropped, 
                (pad_left, pad_right, pad_top, pad_bottom), 
                mode='constant', 
                value=0
            )
        
        # 移除批次维度并调整维度顺序回 (H, W, C)
        result = cropped.squeeze(0).permute(1, 2, 0)
        
        # 转换为 numpy array
        if isinstance(image, np.ndarray):
            result = result.numpy()
        
        return result



    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width_numpy(self, image):
        if self.dynamic_resolution:
            width, height = image.shape[:2]
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width

    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        elif str(file_path).endswith(".json"):
            with open(file_path, "r") as f:
                meta = json.load(f)
            return meta
        elif str(file_path).endswith(".npy"):
            data = np.load(file_path)
            return self.crop_and_resize_numpy(data, *self.get_height_width_numpy(data))
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        data['num_frames'] = self.num_frames
        data['data_id'] = data_id
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat
