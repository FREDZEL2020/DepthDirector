import imageio, os, torch, warnings, torchvision, argparse, json
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import numpy as np
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models.camera import CamVidEncoder
from diffsynth import load_state_dict, save_video

class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        trained_checkpoints=None,
        cpu_offload=False,
    ):
        super().__init__()
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.cpu_offload = cpu_offload
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        print(model_configs)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, redirect_common_files=False)
        self.use_camera_encoder = True
        in_video = 0
        if "depth_cond" in self.extra_inputs:
            in_video = 2
        elif "warp_video" in self.extra_inputs:
            in_video = 2    
        else:
            self.use_camera_encoder = False
        if in_video>0:
            self.pipe.camera_encoder = CamVidEncoder(self.pipe.vae.z_dim, 1024, self.pipe.dit.dim, in_video).to("cpu", dtype=torch.bfloat16)
        
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))

        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")

                if self.use_camera_encoder:
                    state_dict = load_state_dict(lora_checkpoint)
                    key_list = []
                    for key in state_dict.keys():
                        if 'camera_encoder' in key:
                            key_list.append(key)
                    for key in key_list:
                        state_dict[key.replace("pipe.camera_encoder.", "")] = state_dict[key]
                        state_dict.pop(key)

                    missing_keys, unexpected_keys = self.pipe.camera_encoder.load_state_dict(state_dict, strict=False)
                    all_keys = [i for i, _ in self.pipe.camera_encoder.named_parameters()]
                    num_updated_keys = len(all_keys) - len(missing_keys)
                    num_unexpected_keys = len(unexpected_keys)
                    print(f"Camera: {num_updated_keys} parameters are loaded from {lora_checkpoint}. {num_unexpected_keys} parameters are unexpected.")
                # ===== From EX4D ========
            setattr(self.pipe, lora_base_model, model)
            
        if trained_checkpoints is not None:
            state_dict = load_state_dict(trained_checkpoints)
            print(f"Loading trained_checkpoints, total {len(state_dict)} keys")
            missing_keys, unexpected_keys = self.pipe.dit.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in self.pipe.dit.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"DiT: {num_updated_keys} parameters are loaded from {trained_checkpoints}. {num_unexpected_keys} parameters are unexpected.")
            if self.use_camera_encoder:
                key_list = []
                for key in state_dict.keys():
                    if 'camera_encoder' in key:
                        key_list.append(key)
                for key in key_list:
                    state_dict[key.replace("pipe.camera_encoder.", "")] = state_dict[key]
                    state_dict.pop(key)

                missing_keys, unexpected_keys = self.pipe.camera_encoder.load_state_dict(state_dict, strict=False)
                all_keys = [i for i, _ in self.pipe.camera_encoder.named_parameters()]
                num_updated_keys = len(all_keys) - len(missing_keys)
                num_unexpected_keys = len(unexpected_keys)
                print(f"Camera: {num_updated_keys} parameters are loaded from {trained_checkpoints}. {num_unexpected_keys} parameters are unexpected.")
        else:
            print("trained_checkpoints is None")
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
    def forward_preprocess(self, data):
        inputs, inputs_posi = self.prepare_data(data)
        inputs_nega = {}
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs, inputs_posi, inputs_nega)
        return {**inputs, **inputs_posi}

    def prepare_data(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        
        
        # CFG-unsensitive parameters
        inputs = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "height": data["input"][0].size[1],
            "width": data["input"][0].size[0],
            "num_frames": data['num_frames'],
            "use_camera_encoder": self.use_camera_encoder,
            
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        if "video" in data:
            inputs["input_video"] = data["video"]
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                if 'input_image' in data and not isinstance(data["input_image"], str):
                    inputs["input_image"] = data["input_image"][0]
                else:
                    inputs["input_image"] = data["input"][0]
            elif extra_input == "depth_cond":
                inputs["warp_video"] = data["depth"]
                inputs["warp_mask"] = data["mask"]
            elif extra_input == "warp_video":
                inputs["warp_video"] = data["cond"]
                inputs["warp_mask"] = data["mask"]
            elif extra_input== "concat_video":
                inputs["concat_video"] = data["input"]
            else:
                inputs[extra_input] = data[extra_input]
        
        return inputs, inputs_posi
    
    
    def forward(self, data, inputs=None):
        
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss

    # ================ ADD ===============
    @torch.no_grad
    def validate(self, data, output_path, inputs=None):
        inputs, inputs_posi = self.prepare_data(data)
        inputs = {**inputs, **inputs_posi}
        if self.cpu_offload:
            self.pipe.enable_cpu_offload()
        inputs["tiled"] = True
        output_video = self.pipe(**inputs)
        if self.cpu_offload:
            self.pipe.cpu_offload = False
        save_video(output_video, output_path, fps=15, quality=8)

        if "depth" in data:
            save_video(data["depth"], output_path.replace(".mp4", "_depth.mp4"), fps=15, quality=5)
        if "input" in data:
            save_video(data["input"], output_path.replace(".mp4", "_input.mp4"), fps=15, quality=5)
        if "video" in data:
            save_video(data["video"], output_path.replace(".mp4", "_target.mp4"), fps=15, quality=5)

