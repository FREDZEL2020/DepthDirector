import gc
import os
import numpy as np
import torch
from diffusers.training_utils import set_seed
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    sys.path.append(os.path.join(project_root, "3rdparties"))
    sys.path.append(os.path.join(project_root, '3rdparties/GeometryCrafter/third_party/moge'))
    from GeometryCrafter.third_party import MoGe
    from GeometryCrafter.geometrycrafter import (
        GeometryCrafterDiffPipeline,
        GeometryCrafterDetermPipeline,
        PMapAutoencoderKLTemporalDecoder,
        UNetSpatioTemporalConditionModelVid2vid
    )

except Exception as e:
    print(f"Warning: GeometryCrafter not found ({type(e).__name__}: {e})")


class GeometryCrafterDemo:
    def __init__(
        self,
        device: str = "cuda:0",
    ):
        cache_dir="3rdparties/GeometryCrafter/workspace/cache"
        model_type = "diff"
        unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
            'checkpoints/TencentARC/GeometryCrafter',
            subfolder='unet_diff' if model_type == 'diff' else 'unet_determ',
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        ).requires_grad_(False).to(device, dtype=torch.float16)
        self.point_map_vae = PMapAutoencoderKLTemporalDecoder.from_pretrained(
            'checkpoints/TencentARC/GeometryCrafter',
            subfolder='point_map_vae',
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            cache_dir=cache_dir
        ).requires_grad_(False).to(device, dtype=torch.float32)
        self.prior_model = MoGe(
            cache_dir=cache_dir,
        ).requires_grad_(False).to('cuda', dtype=torch.float32)
        if model_type == 'diff':
            self.pipe = GeometryCrafterDiffPipeline.from_pretrained(
                "checkpoints/stabilityai/stable-video-diffusion-img2vid-xt",
                unet=unet,
                torch_dtype=torch.float16,
                variant="fp16",
                cache_dir=cache_dir
            ).to(device)
        else:
            self.pipe = GeometryCrafterDetermPipeline.from_pretrained(
                "checkpoints/stabilityai/stable-video-diffusion-img2vid-xt",
                unet=unet,
                torch_dtype=torch.float16,
                variant="fp16",
                cache_dir=cache_dir
            ).to(device)

        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
            print("Xformers is not enabled")
        # bugs at https://github.com/continue-revolution/sd-webui-animatediff/issues/101
        # self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        frames,
        num_denoising_steps: int,
        guidance_scale: float,
        window_size: int = 110,
        overlap: int = 25,
        seed: int = 42,
        track_time: bool = True,
    ):
        set_seed(seed)
        # inference the depth map using the DepthCrafter pipeline
        with torch.inference_mode():
            rec_point_map, rec_valid_mask = self.pipe(
                frames,
                self.point_map_vae,
                self.prior_model,
                height=576,
                width=1024,
                num_inference_steps=num_denoising_steps,
                guidance_scale=guidance_scale,
                window_size=window_size,
                decode_chunk_size=8,
                overlap=overlap,
                force_projection=True,
                force_fixed_focal=True,
                use_extract_interp=False,
                track_time=track_time,
                low_memory_usage=False
            )


        return rec_point_map, rec_valid_mask
