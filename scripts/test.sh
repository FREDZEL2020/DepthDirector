export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch training/test.py --config $1 \
  --checkpoints $2 --val_dataset_metadata_path $3 --cpu_offload