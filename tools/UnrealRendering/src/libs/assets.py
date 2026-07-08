import unreal
import json
import math
import os
import sys
import random
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(project_root))
from libs.animation import get_anim_properties

def find_actors_skeletalmesh(target_directory, asset_registry):
    """查找所有使用指定骨架的Actor(蓝图)和SkeletalMesh资产"""
    actors = []

    # 查找SkeletalMesh
    skeletal_mesh_filter = unreal.ARFilter(
        class_names=["SkeletalMesh"],
        package_paths=[target_directory],
        recursive_paths=True
    )
    all_skeletal_meshes = asset_registry.get_assets(skeletal_mesh_filter)
    for mesh_asset_data in all_skeletal_meshes:
        mesh_asset = unreal.load_asset(mesh_asset_data.package_name)
        path = str(mesh_asset_data.package_name)

        # 过滤掉不需要的资产
        skip_patterns = ["parts", "attachments", "seperates", "thirdpersoncharacter", "mannequin",
                        "separated", "separates", "clothes", "accessories", "sk_body"]
        
        if any(p in path.lower() for p in skip_patterns):
            continue

        # 检查尺寸是否符合人体
        bounds = mesh_asset.get_bounds()
        box_extent = bounds.box_extent
        height = box_extent.z * 2.0
        width = box_extent.y * 2.0
        if height < 120 or height > 200 or width < 25 or width > 65:
            print(f"过滤掉不符合尺寸要求的SkeletalMesh: {path} (height: {height}, width: {width})")
            continue
        
        if hasattr(mesh_asset, 'skeleton') and mesh_asset.skeleton is not None:
            

            actors.append({
                "package": target_directory,
                "name": str(mesh_asset_data.asset_name),
                "skeletal_mesh_path": path,
                "skeleton_path": str(mesh_asset.skeleton.get_path_name())
            })
    return actors

def find_actors_blueprint(target_directory, asset_registry):
    actors = []
    # 查找Blueprint Actor(蓝图类)
    blueprint_filter = unreal.ARFilter(
        class_names=["Blueprint"],
        package_paths=[target_directory],
        recursive_paths=True
    )
    all_blueprints = asset_registry.get_assets(blueprint_filter)

    for asset in all_blueprints:
        bp_asset_path = asset.package_name
        bp_asset_name = asset.asset_name
        # 加载资产
        blueprint_asset = unreal.load_asset(bp_asset_path)
        path = str(asset.package_name)

        # 过滤蓝图
        skip_patterns = ["Parts", "Attachments", "Seperates", "Weapons",
                        "Separated", "Separates", "Clothes", "Acessories",
                        "BP_Pop", "BP_NPC", "AI_Controller"]
        if any(p in path for p in skip_patterns):
            continue
        if not blueprint_asset:
            print(f"错误: 无法加载蓝图资产 '{bp_asset_path}'")
            return

        # 确保加载的是一个 Blueprint 对象
        if not isinstance(blueprint_asset, unreal.Blueprint):
            print(f"错误: 资产不是一个蓝图对象。")
            return

        # 2. 获取蓝图的生成类 (Generated Class)
        # _C 后缀通常表示已编译的蓝图类
        generated_class = blueprint_asset.generated_class()
        # CDO 包含了蓝图默认组件的设置
        class_default_object = unreal.get_default_object(generated_class)
        # 3. 获取UCharacter类的引用
        # 可以通过静态查找或直接使用 unreal.Character 类
        if not isinstance(class_default_object, unreal.Character):
            continue
        
        # SkeletalMeshComponent -> USkeletalMesh
        skeleton_mesh_comp = class_default_object.mesh
        if hasattr(skeleton_mesh_comp, 'skeletal_mesh'):
            skeleton_mesh = skeleton_mesh_comp.skeletal_mesh
        else:
            continue
        if hasattr(skeleton_mesh, 'skeleton'):
            skeleton_asset_path = class_default_object.mesh.skeletal_mesh.skeleton.get_path_name()
            actors.append({
                "package": target_directory,
                "name": str(asset.asset_name),
                "blueprint_path": path,
                "skeleton_path": str(skeleton_asset_path)
            })


    return actors

def find_animations(target_directory, asset_registry):
    """查找所有属于指定骨架的动画"""
    animations = {}

    anim_sequence_filter = unreal.ARFilter(
        class_names=["AnimSequence"],
        package_paths=[target_directory],
        recursive_paths=True
    )
    all_anim_sequences = asset_registry.get_assets(anim_sequence_filter)

    for anim_asset_data in all_anim_sequences:
        anim_asset = unreal.load_asset(anim_asset_data.package_name)
        props = get_anim_properties(anim_asset)
        if hasattr(anim_asset, 'get_skeleton'):
            try:
                anim_skeleton = anim_asset.get_skeleton()
                if anim_skeleton is not None:
                    anim_skeleton_path = str(anim_skeleton.get_path_name())
                    if anim_skeleton_path not in animations:
                        animations[anim_skeleton_path] = []
                    animations[anim_skeleton_path].append({
                        "name": str(anim_asset_data.asset_name),
                        "path": str(anim_asset_data.package_name),
                        "root_range_cm": props.get("root_range_cm", None),
                    })

                    
                    
            except Exception:
                continue

    return animations

def match_animations(animations):
    """根据动画名称匹配walk、idle和random动画"""
    result = {
        "walk": None,
        "idle": None,
        "random": None
    }

    walk_candidates = []
    idle_candidates = []
    run_candidates = []
    other_candidates = []

    for anim in animations:
        name_lower = anim["name"].lower()
        if "walk" in name_lower:
            # 过滤
            skip_patterns = ["bwd", "backward", "side", "left", "lt", "rt", "right", "oh", "horse"]
            if any(p in name_lower for p in skip_patterns):
                continue
            if anim["root_range_cm"] is not None and anim["root_range_cm"] > 50:
                print(f"过滤掉root移动范围过大的walk动画: {anim['name']} (root_range_cm: {anim['root_range_cm']})")
                continue
            walk_candidates.append(anim)
        elif "run" in name_lower:
            # 过滤
            skip_patterns = ["bwd", "backward", "side", "left", "lt", "rt", "right", "oh", "horse"]
            if any(p in name_lower for p in skip_patterns):
                continue
            if anim["root_range_cm"] is not None and anim["root_range_cm"] > 50:
                print(f"过滤掉root移动范围过大的walk动画: {anim['name']} (root_range_cm: {anim['root_range_cm']})")
                continue
            run_candidates.append(anim)
        elif "idle" in name_lower:
            idle_candidates.append(anim)
        else:
            # 过滤
            skip_patterns = []
            if any(p in name_lower for p in skip_patterns):
                continue
            other_candidates.append(anim)

    # 选择walk动画（优先选择名称中只有walk的）
    if walk_candidates:
        result["walk"] = random.choice(walk_candidates)["path"]

    # if run_candidates:
    #     result["run"] = random.choice(run_candidates)["path"]

    # 选择idle动画（优先选择idle1或只有idle的）
    if idle_candidates:
        result["idle"] = random.choice(idle_candidates)["path"]

    # 选择随机动画（排除walk和idle）
    if other_candidates:
        result["random"] = random.choice(other_candidates)["path"]

    return result

def find_skeleton_related_assets():
    """
    在Unreal Engine项目内容目录中查找所有骨架资源，以及引用它们的骨骼网格体和动画序列。
    返回一个包含这些信息的列表，每个元素是一个字典。
    """
    # 配置项目Content目录的绝对路径（请根据你的实际情况修改）
    # 例如: PROJECT_CONTENT_DIR = r"C:\Users\fredz\Documents\Unreal Projects\我的项目\Content"

    # 初始化Unreal Engine编辑器子系统
    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()

    unreal_base_path = "/Game"
    content_dir = unreal.Paths.project_content_dir()
    print(f"Content物理路径: {content_dir}")

    # 然后你可以使用os.listdir()来列出这个物理路径下的内容
    if os.path.exists(content_dir):
        dir_list = os.listdir(content_dir)

    all_assets_info_list = []  # 存储最终结果的列表
    for package in dir_list:
        
        target_directory = f"/Game/{package}"  # 根据你的项目结构调整路径[2,3](@ref)
        print(f"Search in {target_directory}")
        animations_dict = find_animations(target_directory, asset_registry)
        print(animations_dict.keys())
        skeletal_mesh = find_actors_skeletalmesh(target_directory, asset_registry)
        print("Find SkeletalMesh: ", len(skeletal_mesh))

        blueprints = find_actors_blueprint(target_directory, asset_registry)
        print("Find Blueprint: ", len(blueprints))
        print(animations_dict)
        for sm in skeletal_mesh:
            if sm['skeleton_path'] in animations_dict:
                sm.update({
                    "animation_asset_path": match_animations(animations_dict[sm['skeleton_path']])
                }) 
            all_assets_info_list.append(sm)
        for bp in blueprints:
            if bp['skeleton_path'] in animations_dict:
                bp.update({
                    "animation_asset_path": match_animations(animations_dict[bp['skeleton_path']])
                })
            all_assets_info_list.append(bp)
    return all_assets_info_list

def save_to_json(data, filename):
    """
    将数据保存为JSON文件
    """
    with open(filename, 'w', encoding='utf-8') as f:
        # 使用indent参数美化JSON输出
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到 {filename}")

# 执行主函数
if __name__ == "__main__":
    print("开始扫描项目资产，这可能需要一些时间...")
    result_data = find_skeleton_related_assets()
    save_to_json(result_data,  "D:/UE/UnrealRendering/info/actor.json")
    print(f"共找到 {len(result_data)} 个骨架资源及其关联资产。")