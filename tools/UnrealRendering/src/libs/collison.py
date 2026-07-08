import unreal
import math
import time
import json
import os
import random
import datetime
import re

import importlib
# 1. 导入模块（注意：reload 只能作用于模块对象，不能作用于类）
import params
# 2. 强制刷新这些模块
importlib.reload(params)
# 3. 从刷新后的模块中重新加载具体的类或变量

# ================ Collison Utils =================

def add_box_collision_to_all_meshes():
    # 1. 获取编辑器子系统和库
    editor_filter_lib = unreal.EditorFilterLibrary
    static_mesh_lib = unreal.EditorStaticMeshLibrary
    
    # 2. 获取场景中所有的 StaticMeshActor
    all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
    
    # 3. 筛选出包含 StaticMeshComponent 的 Actor
    # 也可以直接遍历所有 Actor 并检查其组件
    processed_meshes = set() # 用于记录已处理的网格体，避免重复操作
    
    count = 0
    
    for actor in all_actors:
        # 获取 Actor 上的所有静态网格体组件
        components = actor.get_components_by_class(unreal.StaticMeshComponent)
        
        for comp in components:
            mesh = comp.static_mesh
            # 确保网格体存在且没有被处理过
            if mesh and mesh.get_name() not in processed_meshes:
                processed_meshes.add(mesh.get_name())
                collision_count = static_mesh_lib.get_simple_collision_count(mesh)
                bounds = mesh.get_bounding_box()
                z = bounds.max.z - bounds.min.z
                y = bounds.max.y - bounds.min.y
                x = bounds.max.x - bounds.min.x
                if (x < 20 or y < 20 or z < 20) and collision_count < 1:
                    print(f"[INFO][CollisonInit] 网格体 {mesh.get_name()}", collision_count, x, y, z, ", adding collision")
                    # 核心操作：添加盒体简化碰撞
                    # ScriptingCollisionShapeType.BOX 代表盒体
                    static_mesh_lib.add_simple_collisions(mesh, unreal.ScriptingCollisionShapeType.BOX)
                    count += 1
                    print(f"[INFO][CollisonInit] 已为网格体 {mesh.get_name()} 添加盒体碰撞")
                else:
                    continue
            else:
                continue
            
    unreal.EditorLoadingAndSavingUtils.save_dirty_packages(save_map_packages=False, save_content_packages=True)
    print(f"[INFO][CollisonInit] 脚本执行完毕，共处理了 {count} 个唯一的静态网格体资源。")

def force_collision():

    all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
    
    text_label = f"Working on Collison"
    with unreal.ScopedSlowTask(len(all_actors), text_label) as slow_task:
        slow_task.make_dialog(True)               # Makes the dialog visible, if it isn't already
            
        for actor in all_actors:
            if slow_task.should_cancel():         # True if the user has pressed Cancel in the UI
                exit(0)
            slow_task.enter_progress_frame(1)
            is_blueprint = isinstance(actor.get_class(), unreal.BlueprintGeneratedClass)
            if not is_blueprint:
                continue
            if "Roof" not in actor.get_name():
                continue
            # 1. 获取所有物理组件
            primitive_components = actor.get_components_by_class(unreal.PrimitiveComponent)

            for comp in primitive_components:
                # 确保组件开启了碰撞
                comp.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS)
                # 确保它在 Visibility 通道上是 Block 状态
                comp.set_collision_response_to_channel(unreal.CollisionChannel.ECC_VISIBILITY, unreal.CollisionResponseType.ECR_BLOCK)
                
                print(f"[INFO][CollisonInit] 已激活组件碰撞: {comp.get_name()}")
        
    return 

def get_scene_info():
    all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
    count = 0
    bp_count = 0
    text_label = f"Working on Collison"
    # get scene scale
    min_bound = unreal.Vector(0,0,0)
    max_bound = unreal.Vector(0,0,0)

    with unreal.ScopedSlowTask(len(all_actors), text_label) as slow_task:
        slow_task.make_dialog(True)               # Makes the dialog visible, if it isn't already
            
        for actor in all_actors:
            if slow_task.should_cancel():         # True if the user has pressed Cancel in the UI
                exit(0)
            slow_task.enter_progress_frame(1)
            
            is_blueprint = isinstance(actor.get_class(), unreal.BlueprintGeneratedClass)
            if isinstance(actor, unreal.StaticMeshActor) or is_blueprint:
                if not hasattr(actor, "root_component"):
                    continue
                # if 'Wall' not in actor.get_name():
                #     continue
                if actor.root_component is None:
                    continue
                loc = actor.root_component.get_world_location()
                min_bound.x = min(min_bound.x, loc.x)  
                min_bound.y = min(min_bound.y, loc.y)  
                min_bound.z = min(min_bound.z, loc.z)  
                max_bound.x = max(max_bound.x, loc.x)  
                max_bound.y = max(max_bound.y, loc.y)  
                max_bound.z = max(max_bound.z, loc.z) 

                
            if not is_blueprint:
                continue
            # if "Roof" not in actor.get_name():
            #     continue
            # 1. 获取所有物理组件
            primitive_components = actor.get_components_by_class(unreal.PrimitiveComponent)
            procedual = False
            static = False
            spline = False
            for comp in primitive_components:
                if "ProceduralMeshComponent" in comp.get_class().get_name():
                    procedual = True
                if "Spline" in comp.get_class().get_name():
                    spline = True
                if "StaticMeshComponent" in comp.get_class().get_name():
                    static = True
            if not static and procedual:
                count += 1
                print("[INFO][AssetCheck] Invalid Actor", actor.get_name())
            bounds = unreal.SystemLibrary.get_component_bounds(comp)[1]
            if spline and bounds.x > 100 and bounds.y > 100:
                count += 1
                print("[INFO][AssetCheck] Spline",actor.get_name())
            bp_count += 1
        for actor in all_actors:
            if isinstance(actor, unreal.LightmassImportanceVolume):
                center = actor.root_component.get_world_location()
                bounds = unreal.SystemLibrary.get_component_bounds(actor.root_component)[1]
                min_bound = center - bounds #+ unreal.Vector(bounds.x / 2, bounds.y / 2, 0)
                max_bound = center + bounds #- unreal.Vector(bounds.x / 2, bounds.y / 2, 0)
                print("[INFO][AssetCheck] LightImportance Bound", min_bound, max_bound)
                break 
    return {
        "invalid_count": count,
        "bp_count": bp_count,
        "min_bound": min_bound,
        "max_bound": max_bound,
    }


def parse_hit_result(hit, need_actor=False):
    """
    通过 export_text 解析 HitResult 结构体字符串
    """
    if hit is None:
        return {
            "blocking_hit": False,
            "start_penetrating": False,
            "distance": None,
            "location": None,
            "impact_point": None,
            "impact_normal": None,
            "actor_path": None,
            "actor": None
        }
    # 获取原始字符串
    raw_str = hit.export_text()
    
    data = {
        "blocking_hit": "bBlockingHit=True" in raw_str,
        "start_penetrating": "bStartPenetrating=True" in raw_str,
        "distance": 0.0,
        "location": None,
        "impact_point": None,
        "impact_normal": None,
        "actor_path": None,
        "actor": None
    }

    if not data["blocking_hit"]:
        return data

    # 1. 解析数值型属性 (Distance, Time, FaceIndex)
    def get_float(key):
        match = re.search(rf"{key}=([-0-9.]+)", raw_str)
        return float(match.group(1)) if match else 0.0

    data["distance"] = get_float("Distance")

    # 2. 解析向量型属性 (Location, ImpactPoint, Normal, ImpactNormal)
    # 匹配模式：Key=(X=...,Y=...,Z=...)
    def get_vector(key):
        vec_match = re.search(rf"{key}=\(X=([-0-9.]+),Y=([-0-9.]+),Z=([-0-9.]+)\)", raw_str)
        if vec_match:
            return unreal.Vector(
                float(vec_match.group(1)), 
                float(vec_match.group(2)), 
                float(vec_match.group(3))
            )
        return None

    data["location"] = get_vector("Location")
    data["impact_point"] = get_vector("ImpactPoint")
    data["impact_normal"] = get_vector("ImpactNormal")

    # 3. 解析 Actor 信息
    # 在 UE 5.x 中，Actor 信息通常藏在 HitObjectHandle 的 ReferenceObject 里
    actor_match = re.search(r'ReferenceObject="([^"]+)"', raw_str)
    if actor_match:
        full_path = actor_match.group(1)
        # 路径形如: /Game/Maps/Main.Main:PersistentLevel.StaticMeshActor_1.StaticMeshComponent_0
        # 我们通常需要提取 PersistentLevel 之后的 Actor 名称
        data["actor_path"] = full_path
        
        if need_actor:
            # 尝试从路径中获取真正的 Actor 对象
            # 注意：这需要 Actor 当前在场景中已加载
            actor_name = full_path.split('.')[-2] if '.' in full_path else full_path
            data["actor"] = unreal.EditorLevelLibrary.get_all_level_actors().find(actor_name) # 仅作示例，建议用 find_asset_by_path

    return data


def get_hit_location(location, lookat_vector, distance, trace_complex=True):
    world = unreal.UnrealEditorSubsystem().get_editor_world()
    # print(location, lookat_vector, distance)
    hit_result = unreal.SystemLibrary.line_trace_single(
        world,
        location,
        location + lookat_vector.normal() * distance,
        unreal.TraceTypeQuery.TRACE_TYPE_QUERY2,
        trace_complex=trace_complex,
        actors_to_ignore=[],
        draw_debug_type=params.DEBUG_TACE, #PERSISTENT,
        ignore_self=True
    )
    
    if hit_result is not None:
        min_dis = distance
        result = parse_hit_result(hit_result)
        min_dis = min(result['distance'], min_dis)
        return result['distance']
    else:
        # print("No hit")
        return distance + 10



def get_hit_location_sphere(location, lookat_vector, distance, SAFE_VOLUME_RADIUS, trace_complex=True, 
                            trace_type=unreal.TraceTypeQuery.TRACE_TYPE_QUERY2):
    world = unreal.UnrealEditorSubsystem().get_editor_world()
    # print(location, lookat_vector, distance)
    hit_results = unreal.SystemLibrary.sphere_trace_multi(
        world,
        location,
        location + lookat_vector.normal() * distance,
        SAFE_VOLUME_RADIUS,
        trace_type,
        trace_complex=trace_complex,
        actors_to_ignore=[],
        draw_debug_type=params.DEBUG_TACE, #PERSISTENT,
        ignore_self=True
    )
    
    if hit_results is not None:
        min_dis = distance
        for hit_results_item in hit_results:
            result = parse_hit_result(hit_results_item)
            min_dis = min(result['distance'], min_dis)
        return result['distance']
    else:
        # print("No hit")
        return distance + 10


def get_hit_location_double(location, lookat_vector, distance):
    world = unreal.UnrealEditorSubsystem().get_editor_world()
    # print(location, lookat_vector, distance)
    hit_result = unreal.SystemLibrary.line_trace_single(
        world,
        location,
        location + lookat_vector.normal() * distance,
        unreal.TraceTypeQuery.TRACE_TYPE_QUERY2,
        True,               # b_trace_complex
        [],     # actors_to_ignore
        params.DEBUG_TACE, #PERSISTENT,
        True                # b_ignore_self
    )
    hit_result_reverse = unreal.SystemLibrary.line_trace_single(
        world,
        location + lookat_vector.normal() * distance,
        location,
        unreal.TraceTypeQuery.TRACE_TYPE_QUERY2,
        True,               # b_trace_complex
        [],     # actors_to_ignore
        params.DEBUG_TACE, #PERSISTENT,
        True                # b_ignore_self
    )
    hit_result = parse_hit_result(hit_result)
    hit_result_reverse = parse_hit_result(hit_result_reverse)
    # print(f"Double Hit Result: {hit_result} {hit_result_reverse}")
    return hit_result, hit_result_reverse

    
def get_hit_location_sphere_obj(location, lookat_vector, distance, SAFE_VOLUME_RADIUS=100):
    world = unreal.UnrealEditorSubsystem().get_editor_world()
    # print(location, lookat_vector, distance)
    object_types = unreal.Array(unreal.ObjectTypeQuery)
    object_types.append(unreal.ObjectTypeQuery.OBJECT_TYPE_QUERY1)
    object_types.append(unreal.ObjectTypeQuery.OBJECT_TYPE_QUERY2)
    object_types.append(unreal.ObjectTypeQuery.OBJECT_TYPE_QUERY3)
    object_types.append(unreal.ObjectTypeQuery.OBJECT_TYPE_QUERY4)
    object_types.append(unreal.ObjectTypeQuery.OBJECT_TYPE_QUERY5)
    object_types.append(unreal.ObjectTypeQuery.OBJECT_TYPE_QUERY6)
    hit_results = unreal.SystemLibrary.sphere_trace_multi_for_objects(
        world,
        location,
        location + lookat_vector.normal() * distance,
        SAFE_VOLUME_RADIUS,
        object_types=object_types,
        trace_complex=True,
        actors_to_ignore=[],
        draw_debug_type=params.DEBUG_TACE, #PERSISTENT,
        ignore_self=True
    )
    
    if hit_results is not None:
        min_dis = distance
        for hit_results_item in hit_results:
            result = parse_hit_result(hit_results_item)
            min_dis = min(result['distance'], min_dis)
        return result['distance']
    else:
        print("[DEBUG][get_hit_location_sphere_obj] No hit")
        return distance + 10
    
def get_hit_location_obj(location, lookat_vector, distance):
    world = unreal.EditorLevelLibrary.get_editor_world()
    object_types = unreal.Array(unreal.ObjectTypeQuery)
    object_types.append(unreal.ObjectTypeQuery.OBJECT_TYPE_QUERY1)
    object_types.append(unreal.ObjectTypeQuery.OBJECT_TYPE_QUERY2)
    object_types.append(unreal.ObjectTypeQuery.OBJECT_TYPE_QUERY3)
    object_types.append(unreal.ObjectTypeQuery.OBJECT_TYPE_QUERY4)
    object_types.append(unreal.ObjectTypeQuery.OBJECT_TYPE_QUERY5)
    object_types.append(unreal.ObjectTypeQuery.OBJECT_TYPE_QUERY6)
            
    hit_results = unreal.SystemLibrary.line_trace_single_for_objects(
        world,
        location,
        location + lookat_vector.normal() * distance,
        object_types=object_types,
        trace_complex=True,
        actors_to_ignore=[],
        draw_debug_type=params.DEBUG_TACE, #PERSISTENT,
        ignore_self=True
    )
    return hit_results

def check_location_obstacle(location, cam_height, clear_radii_list, SAFE_VOLUME_RADIUS=100):
    world = unreal.EditorLevelLibrary.get_editor_world()
    sample_counts = [2, 3, 6]  # 分别采样1, 6, 12个线段
    has_hit = False
    for radius_idx, radius in enumerate(clear_radii_list):
        sample_count = sample_counts[radius_idx]
        
        # 在圆上均匀采样点
        for segment_idx in range(sample_count):
            # 计算当前线段的起点角度
            start_angle = 2 * math.pi * segment_idx / sample_count
            
            # 计算当前线段的终点角度
            end_angle = 2 * math.pi * (segment_idx + 1) / sample_count
            
            # 计算起点坐标
            start_x = radius * math.cos(start_angle)
            start_y = radius * math.sin(start_angle)
            
            # 计算终点坐标
            end_x = radius * math.cos(end_angle)
            end_y = radius * math.sin(end_angle)
            
            # 进行碰撞检测
            hit_results = unreal.SystemLibrary.sphere_trace_multi(
                world,
                location + unreal.Vector(start_x, start_y, cam_height),  # 起点
                location + unreal.Vector(end_x, end_y, cam_height),      # 终点
                SAFE_VOLUME_RADIUS,
                unreal.TraceTypeQuery.TRACE_TYPE_QUERY2,
                trace_complex=True,
                actors_to_ignore=[],
                draw_debug_type=params.DEBUG_TACE,
                ignore_self=True
            )
                            
            if hit_results is not None and len(hit_results) > 0:
                has_hit = True
                break
        if has_hit:
            break
    if has_hit:
        print("[DEBUG][check_location_obstacle] Exists obstacle around location", location)
        return True
    return  False

