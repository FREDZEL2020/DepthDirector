import os
import time
try:
    import GPUtil
except:
    print("no GPUtil")
from concurrent.futures import ThreadPoolExecutor,as_completed
WORKER_NUM = 5
def system_worker(gpu, job):
    os.system(job)


def dispatch_gpu_jobs(jobs, executor, func, sleep=1, reserved_gpus=[], memory=-1, exclude=True):
    future_to_job = {}
    reserved_gpus = set(reserved_gpus)  # GPUs that are slated for work but may not be active yet
    excluded_gpus = set([])
    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        if memory > 0:
            all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=memory, excludeID=[]))
        else:
            all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=1.0, excludeID=[]))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)
        
        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            
            future = executor.submit(func, gpu, job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)
            if exclude:
                reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing
            time.sleep(sleep)
        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}", flush=True)
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(sleep)

def dispatch_cpu_jobs(jobs, executor, func, sleep=0, reserved_gpus=[]):
    future_to_job = {}
    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        
        # Launch new jobs on available GPUs
        while jobs:
            job = jobs.pop(0)
            
            future = executor.submit(func, -1, job)  # Unpacking job as arguments to worker
            future_to_job[future] = (job)

            time.sleep(sleep)
        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            print(f"Job {job} has finished.", flush=True)
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(sleep)

import json    
from tqdm import tqdm
import uuid
def UE(DATA_ROOT, **kwargs):
    os.makedirs(DATA_ROOT,exist_ok=True)
    # idx_list = [12]
    gpu_jobs = []
    # cmd = f"python recam/run.py --save_depth --task video --dataset {method}    --video_length 81 --device cuda:0 --video_path {input_path} --out_dir {output_path} --raw_dir {raw_dir}"
    print(f"Processing dataset {DATA_ROOT} with task {kwargs['task']}...")
    with ThreadPoolExecutor(max_workers=WORKER_NUM) as executor:
        scenes = os.listdir(DATA_ROOT)
        scenes.sort()
        for file in scenes:
            if os.path.exists(f"{DATA_ROOT}/{file}/videos.txt"):
                continue
            random_uuid = uuid.uuid4()

            cmd = f"cd {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))} && \
                python recam/run.py --device cuda:0 --video_path {DATA_ROOT}/{file} --out_dir /tmp/recam/{random_uuid}/{file} "
            for key in kwargs.keys():
                if key in ["save_depth", "remove_raw", "test_warp", "flex_length"]:
                    if kwargs[key]:
                        cmd += f" --{key}"
                else:
                    cmd += f" --{key} {kwargs[key]}"
            cmd += f" && rsync -av /tmp/recam/{random_uuid}/{file} {DATA_ROOT}/"
            DATA_ROOT_PREFIX = DATA_ROOT.removeprefix("/home/ue_user/code/data/")
            DATA_ROOT_PREFIX = DATA_ROOT_PREFIX.removeprefix("/home/ue_user/data/")
            cmd += f" && ~/bcecmd bos sync {DATA_ROOT} bos:/cdy-video-data/worldreplay/{DATA_ROOT_PREFIX}"
            cmd += f" && python data-pipeline/submit.py --redis_db 75 --dir {DATA_ROOT}/{file}"
            cmd += f" && rm -r /tmp/recam/{random_uuid}"
            gpu_jobs.append(cmd)
            print(f"Added job for scene {file}")
        print(gpu_jobs) 
        dispatch_gpu_jobs(gpu_jobs, executor, system_worker, reserved_gpus=[], exclude=False, memory=0.8)
