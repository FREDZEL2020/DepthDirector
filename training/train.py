import torch, os, json
from diffsynth.trainers.utils import launch_training_task, wan_parser
from diffsynth.trainers.data import VideoDataset
from diffsynth.trainers.logger import ModelLogger
from diffsynth.trainers.trainer import WanTrainingModule
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
from diffsynth.trainers.parser import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script - YAML config version")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args0 = parser.parse_args()
    args = Config.from_yaml(args0.config)
    
    dataset = VideoDataset(args=args)
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
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    print("trainable_modules len:",len(list(model.trainable_modules())))
    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        find_unused_parameters=args.find_unused_parameters,
        num_workers=args.dataset_num_workers,
    )
