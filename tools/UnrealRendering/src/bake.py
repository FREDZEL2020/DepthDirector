import unreal
import os, sys
project_root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(project_root))
from navigator.navigator_navmesh import NavigatorNavMesh

import json
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            args = json.load(f)
    else:
        with open(f"{project_root}/../configs/render_setting_3rd_recam.json", 'r') as f:
            args = json.load(f)
        args.update({"asset_path": "D:/UE/UnrealRendering/info/assets.json",})
        
    navigator = NavigatorNavMesh(args['navigator'], args['collison'])
    navigator.get_world_bound()
    navigator.build_navigation(navigator.min_bound, navigator.max_bound)
    navigator.setup_navigation()
    world = unreal.EditorLevelLibrary.get_editor_world()
    current_level_path = world.get_path_name().split('.')[0] # 去掉后缀 .LevelName
    unreal.log(f"当前关卡路径: {current_level_path}")
    unreal.EditorLevelLibrary.save_current_level()