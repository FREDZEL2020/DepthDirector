import unreal
import json
import os

def export_level_paths_to_json():
    # 1. 获取当前脚本所在的目录
    # 注意：如果在虚幻编辑器内直接粘贴运行（没有保存为文件），__file__ 可能会报错
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # 如果是在编辑器直接粘贴运行的，退而求其次保存在项目根目录
        current_dir = unreal.Paths.project_dir()
        unreal.log_warning("脚本未保存为文件，JSON 将导出到项目根目录。")

    output_path = os.path.join(current_dir, "LevelPaths.json")

    # 2. 获取资产注册表并设置过滤
    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()
    filter = unreal.ARFilter(class_names=["World"], recursive_paths=True)
    asset_data_list = asset_registry.get_assets(filter)
    
    # 3. 提取路径：排除掉名称中包含 overview 的资产
    path_list = [
        str(asset.package_name) for asset in asset_data_list 
        if "overview" not in str(asset.asset_name).lower() and "sublevel" not in str(asset.package_name).lower() and "blueprint" not in str(asset.package_name).lower()
    ]

    # 4. 写入 JSON
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(path_list, f, indent=4, ensure_ascii=False)
        unreal.log(f"成功导出 {len(path_list)} 条路径至: {output_path}")
    except Exception as e:
        unreal.log_error(f"写入文件失败: {str(e)}")

# 执行
export_level_paths_to_json()