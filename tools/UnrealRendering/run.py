import os
import sys
import json
import time
project_root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(project_root)
import subprocess
from utils.redis import REDIS    
from utils.task import run_cook, run_prerender_ddc, run_render, check_ps, build_template, run_render_ddc, run_navmesh
import argparse
from utils.storage import get_storage_instance
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from utils.worker import UE
class Runner:
    def __init__(self, args):
        self.args = args
        DB_NUM = args.db_num if args.db_num is not None else 0
        REDIS_URL = args.redis_url if args.redis_url is not None else f"redis://{os.environ.get('REDIS_URL', 'redis.itjdzririeic.scs.bj.baidubce.com')}:24092"
        REDIS_PASSWD = args.redis_passwd if args.redis_passwd is not None else "Unkwown"
        self.redis_inst = REDIS(
            REDIS_URL,
            REDIS_PASSWD,
            TO_BE_PROCESSED_KEY=f"to-be-processed",
            FAILED_KEY=f"failed",
            DONE_KEY=f"finished",
            BUILT_KEY="built",
            db=DB_NUM
        )
        self.height = 720
        self.width = 1280

    def get_scene(self):
        
        data = self.redis_inst.pop(1)
        if len(data) == 0:
            print("db empty, finished")
            time.sleep(10)
            exit(0)
        # data = ["/Game/StylizedVikingFort/Levels/Demonstration_Level"]
        print("Getting:", data)
        scene_data = json.loads(data[0])
        scene_path = scene_data["level_path"]
        items = str(scene_path).rstrip("/").rstrip(".umap").split("/")
        scene_name = items[2]
        level_name = items[-1]
        if "raw_path" in scene_data:
            raw_path = scene_data["raw_path"]
        else:
            raw_path = scene_name
        
        return scene_name, level_name, scene_path, raw_path
    
    def get_built_scene(self):
        
        data = self.redis_inst.redis_inst.spop(self.redis_inst.BUILT_KEY, 1)
        if len(data) == 0:
            print("db empty, finished", flush=True)
            time.sleep(10)
            return None,None,None
        # data = ["/Game/StylizedVikingFort/Levels/Demonstration_Level"]
        print("Getting:", data)
        scene_data = json.loads(data[0])
        scene_path = scene_data["level_path"]
        items = scene_path.split("/")
        scene_name = items[2]
        level_name = items[-1]
        if "raw_path" in scene_data:
            raw_path = scene_data["raw_path"]
        else:
            raw_path = scene_name
        
        
        return scene_name, level_name, scene_path, raw_path
    
    def submit(self):
        print("Submiting")
        with open(self.args.levels_json) as f:
            scene_list = json.load(f)
        self.redis_inst.submit(scene_list)



    def cook(self):
        project_path = os.path.join(self.args.project_path, "COOK")
        os.makedirs(self.args.logs_path, exist_ok=True)
        if project_path.startswith("/data/"):
            build_template(project_path)
            
        
        self.storage = get_storage_instance(args.storage['storage_type'], project_path, args.storage)
        self.storage.load_project("COOK")

        # generate base shaders
        if os.path.exists(f"{project_path}/ddc"):
            os.system(f"rsync -aqL {project_path}/ddc /data/")
        else:
            null_script = os.path.join(project_root, "src", "null.py")
            run_prerender_ddc(self.args.unreal_engine_path, project_path, self.args.logs_path, null_script, "COOK", "NONE", "")
        # os.system(f"cd /data && find ddc -type f > exclude_list.txt")
        run_cook(self.args.unreal_engine_path, project_path, self.args.logs_path, "COOK", "NONE", "")
        self.storage.save_ddc()

    def build(self):
        scene_name, level_name, scene_path, raw_path = self.get_scene()
        if os.path.exists(f"{self.args.output_path}/{scene_name}/"):
            print(f"allready processed at {self.args.output_path}/{scene_name}/, finished")
            return
        else:
            print("Output to", f"{self.args.output_path}/{scene_name}/")
        project_path = self.args.project_path
        os.makedirs(self.args.logs_path, exist_ok=True)
        if project_path.startswith("/data/"):
            project_path = os.path.join(self.args.project_path, scene_name)
            
            build_template(project_path)
            
        self.storage = get_storage_instance(args.storage['storage_type'], project_path, args.storage)
           
        self.storage.load_project(raw_path)
        os.system(f"rsync -aqL {project_path}/ddc /data/")
        # self.storage.load_ddc(scene_name)
        config = {
            "output_path": self.args.output_path,
            "raw_output_path": self.args.raw_output_path,
            "dataset_type": self.args.dataset_type,
            "sequence_name": f"{scene_name}_{level_name}",
            "animation_path": os.path.join(project_root, "info", "anims.json"),
            "character_path": os.path.join(project_root, "info", "actor.json"),
            "height": self.height,
            "width": self.width,
        }
        if self.args.car:
            print("Use car actor")
            config["character_path"] = os.path.join(project_root, "info", "car_actor.json")
        with open(self.args.render_settings, "r") as f:
            render_settings = json.load(f)
        config.update(render_settings)
        with open(f"/tmp/{scene_name}_{level_name}.json", 'w') as f:
            json.dump(config, f, indent=1)
        
        # put navmesh first
        if "i2v" == render_settings["renderer"]:
            pass
        else:
            python_script = f"{self.args.render_code}/bake.py {f'/tmp/{scene_name}_{level_name}.json'}"
            if os.path.exists("/data/ddc"):
                print("Use ddc")
                run_render_ddc(self.args.unreal_engine_path, project_path, self.args.logs_path, python_script, scene_name, level_name, scene_path, self.args.display)
            else:
                run_render(self.args.unreal_engine_path, project_path, self.args.logs_path, python_script, scene_name, level_name, scene_path, self.args.display)
        
        # build sequence
        python_script = f"{self.args.render_code}/build.py {f'/tmp/{scene_name}_{level_name}.json'}"
        if os.path.exists("/data/ddc"):
            print("Use ddc")
            run_render_ddc(self.args.unreal_engine_path, project_path, self.args.logs_path, python_script, scene_name, level_name, scene_path, self.args.display)
        else:
            run_render(self.args.unreal_engine_path, project_path, self.args.logs_path, python_script, scene_name, level_name, scene_path, self.args.display)
        with open(f"/tmp/{scene_name}_{level_name}.json", 'r') as f:
            results = json.load(f)
        success = results.get('results')
        if success:
            self.redis_inst.add(scene_path, status="built")
        else:
            self.redis_inst.add(scene_path, status="fail")
        # os.system(f"rm -r {project_path}")
        return project_path, scene_name, level_name, scene_path
    
    def build_and_render(self):
        project_path, scene_name, level_name, scene_path = self.build()
        if os.path.exists(f"{self.args.output_path}/{scene_name}/"):
            print(f"allready processed at {self.args.output_path}/{scene_name}/, finished")
            return
        else:
            print("Output to", f"{self.args.output_path}/{scene_name}/")
        self._render(project_path, scene_name, level_name, scene_path)

    def render(self):
        scene_name, level_name, scene_path, raw_path = self.get_built_scene()
        if scene_name is None:
            return
        if os.path.exists(f"{self.args.output_path}/{scene_name}/"):
            print(f"allready processed at {self.args.output_path}/{scene_name}/, finished")
            return
        else:
            print("Output to", f"{self.args.output_path}/{scene_name}/")
        project_path = self.args.project_path
        os.makedirs(self.args.logs_path, exist_ok=True)
        if project_path.startswith("/data/"):
            project_path = os.path.join(self.args.project_path, scene_name)
            
            build_template(project_path)
            
        self.storage = get_storage_instance(args.storage['storage_type'], project_path, args.storage)
           
        self.storage.load_project(raw_path)
        os.system(f"rsync -aqL {project_path}/ddc /data/")
        # self.storage.load_ddc(scene_name)
        self._render(project_path, scene_name, level_name, scene_path)
    
    def _render(self, project_path, scene_name, level_name, scene_path):
        config = {
            "output_path": self.args.output_path,
            "raw_output_path": self.args.raw_output_path,
            "dataset_type": self.args.dataset_type,
            "sequence_name": f"{scene_name}_{level_name}",
            "animation_path": os.path.join(project_root, "info", "anims.json"),
            "character_path": os.path.join(project_root, "info", "actor.json"),
            "height": self.height,
            "width": self.width,
        }
        if self.args.car:
            print("Use car actor")
            config["character_path"] = os.path.join(project_root, "info", "car_actor.json")
        with open(self.args.render_settings, "r") as f:
            render_settings = json.load(f)
        config.update(render_settings)
        with open(f"/tmp/{scene_name}_{level_name}.json", 'w') as f:
            json.dump(config, f, indent=1)
        python_script = f"{self.args.render_code}/render.py {f'/tmp/{scene_name}_{level_name}.json'}"
        if os.path.exists("/data/ddc"):
            print("Use ddc")
            run_render_ddc(self.args.unreal_engine_path, project_path, self.args.logs_path, python_script, scene_name, level_name, scene_path, self.args.display)
        else:
            run_render(self.args.unreal_engine_path, project_path, self.args.logs_path, python_script, scene_name, level_name, scene_path, self.args.display)
        time.sleep(10)
        pod_name = os.environ.get('POD_NAME', 'Pod')
        log_file_path = os.path.join(self.args.logs_path, f'postprocess_{scene_name}_{level_name}_{pod_name}.log')
        
        print(f"Start Postprocess. Logs to {log_file_path}", flush=True)
        # log_stream = open(log_file_path, "w")
        kwargs = {}
        if self.args.depth_downsample is not None:
            kwargs["save_depth"] = True,
            kwargs["depth_downsample_rate"] = self.args.depth_downsample
            kwargs["flex_length"] = True
        results=UE(DATA_ROOT=f"{self.args.output_path}/{scene_name}_{level_name}", 
            raw_dir=self.args.raw_output_path,
            video_length=render_settings['navigator']['frame_num']-4,
            dataset=self.args.dataset_type,
            task="process",
            remove_raw=True,
            sample_size=f"{self.height} {self.width}",
            **kwargs
        )
        print(results)
        while True:
            print("Loop, check if UnrealEditor exists", flush=True)
            if not check_ps("recam/run.py")[0]:
                break
            time.sleep(60)
        self.redis_inst.add(scene_path, status="success")
        os.system(f"rm -r {self.args.raw_output_path}/{scene_name}_{level_name}")
        os.system(f"rm -r {project_path}")
        os.system(f"rm -r /data/ddc")

    def navmesh(self):
        scene_name, level_name, scene_path, raw_path = self.get_scene()
        project_path = self.args.project_path
        os.makedirs(self.args.logs_path, exist_ok=True)
        if project_path.startswith("/data/"):
            project_path = os.path.join(self.args.project_path, scene_name)
            
            build_template(project_path)
            
        self.storage = get_storage_instance(args.storage['storage_type'], project_path, args.storage)
           
        self.storage.load_project(raw_path)
        os.system(f"rsync -aqL {project_path}/ddc /data/")
        
        config = {
            "output_path": self.args.output_path,
            "raw_output_path": self.args.raw_output_path,
            "dataset_type": self.args.dataset_type,
            "sequence_name": f"{scene_name}_{level_name}",
            "animation_path": os.path.join(project_root, "info", "anims.json"),
            "character_path": os.path.join(project_root, "info", "actor.json"),
        }
        with open(self.args.render_settings, "r") as f:
            render_settings = json.load(f)
        config.update(render_settings)
        with open(f"/tmp/{scene_name}_{level_name}.json", 'w') as f:
            json.dump(config, f, indent=1)
        python_script = f"{self.args.render_code}/bake.py {f'/tmp/{scene_name}_{level_name}.json'}"
        run_render_ddc(self.args.unreal_engine_path, project_path, self.args.logs_path, python_script, scene_name, level_name, scene_path, self.args.display)
        time.sleep(10)
        pod_name = os.environ.get('POD_NAME', 'Pod')
        log_file_path = os.path.join(self.args.logs_path, f'postprocess_{scene_name}_{level_name}_{pod_name}.log')
        os.system(f"rm -r {project_path}")

    def build_loop(self):
        while True:
            self.build()

    def full_loop(self):
        while True:
            self.build_and_render()

    def render_loop(self):
        while True:
            self.render()

    def navmesh_loop(self):
        while True:
            self.navmesh()

    def cook_loop(self):
        while True:
            os.system(f"rm -r /data/ddc")
            self.cook()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, default=None,
                        help='JSON config file to load options from.')
    parser.add_argument('--storage', type=str, required=False, default=None,
                        help='JSON config file to load options from.')
    
    # parse_known_args 会返回已知参数（如--config）和其余未知参数
    args, argv = parser.parse_known_args()

    # 如果指定了配置文件，则加载它
    if args.config is not None:
        with open(args.config, 'rt') as f:
            config_dict = json.load(f)
        # 将JSON配置更新到args的命名空间中
        args.__dict__.update(config_dict)
        print(f"Config loaded from {args.config}.")

    if args.storage is not None:
        with open(args.storage, 'rt') as f:
            config_dict = json.load(f)
        print(f"Config loaded from {args.storage}.")
        # 将JSON配置更新到args的命名空间中
        args.__dict__.update({"storage": config_dict})
        

    # 第二阶段：定义所有完整的参数
    # 注意：add_argument 时设置的 default 值，只有在命令行和JSON中都没有该参数时才生效
    parser.add_argument('--render_settings', type=str, default="configs/render_settings.json")
    parser.add_argument('--unreal_engine_path', type=str, default=None)
    parser.add_argument('--project_path', type=str, default=None)
    parser.add_argument('--render_code', type=str, default=None)
    parser.add_argument('--post_process_code', type=str, default=None)
    parser.add_argument('--dataset_type', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--raw_output_path', type=str, default=None)
    parser.add_argument('--levels_json', type=str, default=None)
    parser.add_argument('--logs_path', type=str, default=None)
    parser.add_argument('--depth_downsample', type=str, default=None)
    # redis 相关参数
    parser.add_argument('--db_num', type=int, default=None)
    parser.add_argument('--redis_url', type=str, default=None)
    parser.add_argument('--redis_passwd', type=str, default=None)

    parser.add_argument('--display', action='store_true')
    parser.add_argument('--car', action='store_true')
    
    parser.add_argument('--task', type=str, default='cook', help='Mode: cook or render or submit')
    # 第三阶段：再次解析命令行（包括之前未知的参数argv），并更新args
    # 这一步会让命令行参数覆盖JSON配置文件中的同名参数
    args = parser.parse_args(args=argv, namespace=args)

    runner = Runner(args)
    getattr(runner, args.task)()