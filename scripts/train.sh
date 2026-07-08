export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

accelerate launch training/train.py --config $1
