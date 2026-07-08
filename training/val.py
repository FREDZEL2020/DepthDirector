import torch, os, json
from diffsynth.trainers.trainer import WanTrainingModule
from diffsynth.trainers.data import VideoDataset
from diffsynth.trainers.logger import ModelLogger
from diffsynth.trainers.trainer import WanTrainingModule
from diffsynth.trainers.parser import Config

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script - YAML config version")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoints", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")

    args0 = parser.parse_args()
    args = Config.from_yaml(args0.config)
    if args0.checkpoints is not None:
        if args.lora_base_model is None:
            args.trained_checkpoints = os.path.join(args.output_path, args0.checkpoints+".safetensors")
        else:
            args.lora_checkpoint = os.path.join(args.output_path, args0.checkpoints+".safetensors")
    
    dataset = VideoDataset(
            base_path=args.val_dataset_base_path,
            metadata_path = args.val_dataset_metadata_path,
            height = args.height,
            width = args.width,
            max_pixels = args.max_pixels,
            num_frames = args.num_frames,
            data_file_keys = args.data_file_keys.split(","),
            repeat = 1
        )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        trained_checkpoints=args.trained_checkpoints
    )
    output_path = f"{args.output_path}/{args0.checkpoints.replace('-','_')}_val"
    os.makedirs(output_path, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0])
    model, dataloader = accelerator.prepare(model, dataloader)

    # for i, data in enumerate(dataloader):
    for i in range(len(dataloader)):
        print("VALIDATING: ", i*accelerator.num_processes+accelerator.process_index, "/", len(dataset), " samples")
        data = dataset.__getitem__(i*accelerator.num_processes+accelerator.process_index)
        print(data.keys())
        if os.path.exists(f"{output_path}/video{i*accelerator.num_processes+accelerator.process_index}.mp4"):
            continue
        with torch.no_grad():
            if isinstance(model,WanTrainingModule):
                model.validate(data, f"{output_path}/video{i*accelerator.num_processes+accelerator.process_index}.mp4")
            else:
                model.module.validate(data, f"{output_path}/video{i*accelerator.num_processes+accelerator.process_index}.mp4")
        