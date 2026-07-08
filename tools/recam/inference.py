from processor.trajectory import TrajectoryProcessor
import torch
import os
import argparse
torch.backends.cuda.preferred_linalg_library('magma')

def get_parser():
    parser = argparse.ArgumentParser()

    ## general
    parser.add_argument('--video_path', type=str, help='Input path')    
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
        '--traj_txt',
        type=str,
        default=None,
        help="Required for 'traj' camera, a txt file that specify camera trajectory",
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        nargs=2,
        default=[720, 1280],
        help='Sample size as [height, width]',
    )
    parser.add_argument('--blip_path', type=str, default="checkpoints/Salesforce/blip2-opt-2.7b")
    parser.add_argument('--save_depth', action="store_true", default=False)
    parser.add_argument('--metadata', action="store_true", default=False)
    
    parser.add_argument('--fps', type=int, default=10, help='Fps for saved video')
    parser.add_argument(
        '--geometry',
        type=str,
        default="Dynamic3DMesh",
        help='task name',
    )
    parser.add_argument(
        '--task',
        type=str,
        default="process",
        help='task name',
    )
    parser.add_argument(
        '--depth_model',
        type=str,
        default="",
        help='method name',
    )
    return parser


def submit(opts):
    from utils.redis import REDIS
    redis_inst = REDIS(
        TO_BE_PROCESSED_KEY=f"to-be-processed",
        FAILED_KEY=f"failed",
        DONE_KEY=f"finished"
    )
    input_list = []
    if os.path.isdir(opts.video_path):
        for root, dirs, files in os.walk(opts.video_path):
            # 检查当前目录下是否存在
            for file in files:
                if file.endswith("mp4"):
                    input_list.append([os.path.join(root,file), f"{os.path.basename(root)}_{file.removesuffix('.mp4')}"])
    else:
        input_list.append([opts.video_path, opts.video_path.removesuffix('.mp4')])
    redis_inst.submit(input_list)

if __name__ == "__main__":
    parser = get_parser()  # infer config.py
    opts = parser.parse_args()
    opts.weight_dtype = torch.bfloat16
    
    opts.save_dir = opts.out_dir
    os.makedirs(opts.save_dir, exist_ok=True)
    print(opts.geometry, opts.depth_model, opts.task)
    if opts.task=="submit":
        submit(opts)
        exit(0)
    pvd = TrajectoryProcessor(opts)
    getattr(pvd, opts.task)()