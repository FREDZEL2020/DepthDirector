from processor import PROCESSORS
import torch
import os
import argparse
torch.backends.cuda.preferred_linalg_library('magma')

def get_parser():
    parser = argparse.ArgumentParser()

    ## general
    parser.add_argument('--video_path', type=str, help='Input path')
    parser.add_argument('--raw_dir', type=str, help='Input raw path')
    
    parser.add_argument('--image_path', type=str, help='Input path')
    parser.add_argument(
        '--out_dir', type=str, default='./experiments/', help='Output dir'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='The device to use'
    )
    parser.add_argument(
        '--video_length', type=int, default=81, help='Length of the video frames'
    )
    parser.add_argument(
        '--K_scale', type=float, default=1.0, help='Scale factor for K matrix'
    )
    parser.add_argument('--save_depth', action="store_true", default=False)
    parser.add_argument('--test_warp', action="store_true", default=False)
    parser.add_argument('--remove_raw', action="store_true", default=False)
    parser.add_argument('--flex_length', action="store_true", default=False, help="Override video_length, use frame number in camera json")
    
    parser.add_argument(
        '--depth_downsample_rate',
        type=int,
        nargs=3,
        default=[1,1,1],
        help='Sample size as [height, width]',
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        nargs=2,
        default=[720, 1280],
        help='Sample size as [height, width]',
    )
    parser.add_argument('--blip_path', type=str, default="checkpoints/blip2-opt-2.7b")
    parser.add_argument('--fps', type=int, default=30, help='Fps for saved video')
    parser.add_argument(
        '--task',
        type=str,
        default="process",
        help='task name',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default="UE_Static",
        help='method name',
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()  # infer config.py
    opts = parser.parse_args()
    opts.weight_dtype = torch.bfloat16
    
    opts.save_dir = opts.out_dir
    os.makedirs(opts.save_dir, exist_ok=True)
    print(opts.dataset, opts.task)
    
    if f"{opts.dataset}" in PROCESSORS:
        pvd = PROCESSORS[f"{opts.dataset}"](opts)
    else:
        assert False
    getattr(pvd, opts.task)()