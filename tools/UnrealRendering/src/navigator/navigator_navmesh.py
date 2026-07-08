import unreal
import math
import time
import json
import os
import random
import datetime

import importlib
# 1. 导入模块（注意：reload 只能作用于模块对象，不能作用于类）
import params
import navigator.navigator
import libs.collison as lib_collison
import libs.math as lib_math

# 2. 强制刷新这些模块
importlib.reload(params)
importlib.reload(navigator.navigator)
importlib.reload(lib_collison)
# 3. 从刷新后的模块中重新加载具体的类或变量
from navigator.navigator import Navigator

class NavigatorNavMesh(Navigator):
    def __init__(self, navigator_config, collison_config):
        super().__init__(navigator_config, collison_config)
        self.camera_type = [
            'navigate'
        ]
        world = unreal.EditorLevelLibrary.get_editor_world()
        # NOTE: override collison config for navmesh, use small distance
        self.navigator_config['forward_collison'] = {
            "distance": 300,
            "height": 100
        }
        self.nav_data = unreal.NavigationSystemV1.get_navigation_system(world)
        self.decompose_RT = True
    
    def setup(self):
        super().setup()
        # self.build_navigation(self.min_bound, self.max_bound)
        self.setup_navigation()
        world = unreal.EditorLevelLibrary.get_editor_world()
        current_level_path = world.get_path_name().split('.')[0] # 去掉后缀 .LevelName
        unreal.log(f"当前关卡路径: {current_level_path} 保存NavMesh")
        unreal.EditorLevelLibrary.save_current_level()

    
    def setup_navigation(self):
        editor_world = unreal.EditorLevelLibrary.get_editor_world()
        unreal.log(f"Rebuild")
        unreal.SystemLibrary.execute_console_command(editor_world, "RebuildNavigation")
        unreal.SystemLibrary.execute_console_command(editor_world, "RebuildAll")
        unreal.log(f"Rebuild Waited")
        
    def build_navigation(self, min_bound, max_bound):
        self.destroy_all_block_volume()
        all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
        for actor in all_actors:
            # if isinstance(actor, unreal.CineCameraActor):
            #     unreal.EditorLevelLibrary.destroy_actor(actor)
            # if isinstance(actor, unreal.SkeletalMeshActor):
            #     unreal.EditorLevelLibrary.destroy_actor(actor)
            if isinstance(actor, unreal.NavMeshBoundsVolume):
                unreal.EditorLevelLibrary.destroy_actor(actor)
        
        navmesh_class = unreal.NavMeshBoundsVolume
        location = (min_bound + max_bound ) /2
        rotation = unreal.Rotator(0.0, 0.0, 0.0)
        print("[INFO][Navigator] min_bound", min_bound)
        print("[INFO][Navigator] max_bound", max_bound)

        new_navmesh_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
            navmesh_class, 
            location, 
            rotation
        )
        if new_navmesh_actor is not None:
            unreal.log("成功放置 NavMesh Bounds Volume。")

            # ----------------------------------------------------------------------
            # 步骤 5: 设置 Actor 的缩放 (Bounds)
            # ----------------------------------------------------------------------
            # NavMesh Bounds Volume 的范围是通过其 Transform 组件的 Scale 属性来控制的。
            # 假设我们想要一个 X=20000, Y=20000, Z=1000 的大区域 (即 X=20m, Y=20m, Z=1m)
            scale_vector = ( max_bound - min_bound ) / 100 # 注意：在UE中，Transform Scale的 1.0 表示 100cm。
            scale_vector.z = min(100, scale_vector.z)
            scale_vector.x = min(scale_vector.x, 1000)
            scale_vector.y = min(scale_vector.y, 1000)
            print("[INFO][Navigator] navmesh location", location)
            print("[INFO][Navigator] navmesh scale_vector", scale_vector) # 135 116
            # 设置根组件的相对缩放
            new_navmesh_actor.root_component.set_relative_scale3d(scale_vector)
            
            print(f"[INFO][Navigator] NavMesh Bounds Volume 已放置于 {location}，缩放设置为 {scale_vector}。")
    

    def project_to_nav(self, world, point: unreal.Vector) -> unreal.Vector:
        point = unreal.NavigationSystemV1.project_point_to_navigation(world, point, None, None)
        return point

    def get_location_to_ground(self, location):
        world = unreal.EditorLevelLibrary.get_editor_world()
        hit_distance_list = self.test_navigable_area(location)
        if all([d < 100 for d in hit_distance_list]):
            return super().get_location_to_ground(location)
        location = self.project_to_nav(world, location)
        return True, location
    
    def test_ground_for_points(self, ground_location_list, test_start_height=200, max_hit_distance=1000,
                    min_height=-200, max_height=200):
        world = unreal.EditorLevelLibrary.get_editor_world()
        hit_distance_list = self.test_navigable_area(ground_location_list[0])
        if all([d < 100 for d in hit_distance_list]):
            print("[DEBUG][RandomWalk Navigate NavMesh] test_ground_for_points, NavMesh not available or too small area, use original method")
            return super().test_ground_for_points(ground_location_list, test_start_height, max_hit_distance, min_height, max_height)
        
        valid_list = []
        height_list = []

        for idx, location in enumerate(ground_location_list):
            ground_location = self.project_to_nav(world, location)
            height = location.z - ground_location.z
            height_list.append(height)
            valid_list.append(True)
        return valid_list, height_list
    
    def test_navigable_area(self, location, test_range=2000, sample_number=12):
        world = unreal.EditorLevelLibrary.get_editor_world()

        project_location = self.project_to_nav(world, location)
        if project_location is None:
            return [0]
        print("[DEBUG][RandomWalk Navigate NavMesh] test_navigable_area, location project, ", location, project_location)
        hit_distance_list = []
        for segment_idx in range(sample_number):
            # 计算当前线段的起点角度
            start_angle = 2 * math.pi * segment_idx / sample_number

            # 计算终点坐标
            end_x = test_range * math.cos(start_angle)
            end_y = test_range * math.sin(start_angle)
            
            hit_location = unreal.NavigationSystemV1.navigation_raycast(
                world_context_object=world,
                ray_start=project_location,
                ray_end=project_location+unreal.Vector(end_x, end_y, 0),
                filter_class=None,
                querier=None,
            )
            if hit_location is None:
                distance = 2000
            else:
                distance = (hit_location-project_location).length()
            hit_distance_list.append(distance)
        print("[DEBUG][RandomWalk Navigate NavMesh] test_navigable_area", hit_distance_list)
        return hit_distance_list
    
    def get_hit_location_on_nav(self, location, direction, distance=2000):
        world = unreal.EditorLevelLibrary.get_editor_world()
        hit_location = unreal.NavigationSystemV1.navigation_raycast(
            world_context_object=world,
            ray_start=location,
            ray_end=location + direction * distance,
            filter_class=None,
            querier=None,
        )

        # Determine actual travel distance
        if hit_location is None:
            # Path is clear
            return distance + 10
        else:
            # Obstacle detected
            hit_dist = (location - hit_location).length()
            return hit_dist

    

    def surrounding_collision(self, ground_location, distance=[0,50], height=[0,180]):
        """
        Note that: height+self.SAFE_VOLUME_RADIUS
        """
        hit_distance_list = self.test_navigable_area(ground_location)
        if all([d > distance[-1] for d in hit_distance_list]):
            print("[DEBUG][Navmesh Surrounding Collision] OK location", ground_location, "hit_distance_list, ", hit_distance_list)
            return False
        else:
            print("[DEBUG][Navmesh Surrounding Collision] FAIL location", ground_location, "hit_distance_list, ", hit_distance_list)
            return True

    def get_random_start(self, start_point_collison_config):
        """
        start_point_collison: dict
        """
        if self.playerstart is None:
            return super().get_random_start(start_point_collison_config)
        world = unreal.EditorLevelLibrary.get_editor_world()
        i = 0
        while i < params.MAX_ATTEMPTS:
            # use different types of actor as search anchor
            origin = self.playerstart.get_actor_location()
            nav_location = unreal.NavigationSystemV1.get_random_reachable_point_in_radius(world, origin, (self.max_bound-self.min_bound).length(), None)
            print("[INFO][Navmesh get_random_start] Trying random location, ", i, origin, nav_location)
            i += 1
            if self.check_location(nav_location, nav_rotator=None, **start_point_collison_config):
                print("[SUCCESS][Navmesh get_random_start] Get perfect camera: ", nav_location)
                return nav_location
        print("[ERROR][Navmesh get_random_start] Fail to get valid camera")
        return None

    def get_random_start_and_direction(self, start_point_collison_config):
        """
        start_point_collison: dict
        """
        world = unreal.EditorLevelLibrary.get_editor_world()
        i = 0
        while i < params.MAX_ATTEMPTS:
            # use different types of actor as search anchor
            if self.playerstart is not None:
                origin = self.playerstart.get_actor_location()
            else:
                random.seed(time.time())
                actor_level = random.randint(0, len(self.all_level_actors)-1)
                actor = self.all_level_actors[actor_level][random.randint(0, len(self.all_level_actors[actor_level])-1)]
                if actor is None or actor.root_component is None:
                    print("[ERROR][Navmesh get_random_start_and_direction] Invalid actor", actor)
                    continue
                origin, box_extent = actor.get_actor_bounds(only_colliding_components=False)
                print("[INFO][Navmesh get_random_start_and_direction] Use actor", actor.get_actor_label())
            nav_location = unreal.NavigationSystemV1.get_random_reachable_point_in_radius(world, origin, (self.max_bound-self.min_bound).length(), None)
            if nav_location is None:
                print("[FAIL][Navmesh get_random_start_and_direction] Failed to get random reachable point in radius, ", origin)
                continue
            print("[INFO][Navmesh get_random_start_and_direction] Trying random location, ", i, origin, nav_location)
            i += 1
            
            if self.check_location(nav_location, nav_rotator=None, **start_point_collison_config):
                print("[SUCCESS][Navmesh get_random_start_and_direction] Get perfect camera: ", nav_location)
                try:
                    nav_rotator = self.get_valid_rotation(nav_location, unreal.Vector(0,0,0), 180,
                                        start_point_collison_config.get("forward_collison",None),
                                        start_point_collison_config.get("backward_collison",None),
                                        start_point_collison_config.get("forward_collision_with_width",None),
                                        start_point_collison_config.get("forward_slope",None))
                except Exception as e:
                    print("[FAIL][Navmesh get_random_start_and_direction] Failed to get valid rotation, ", e)
                    continue
                return nav_location, nav_rotator
        print("[ERROR][Navmesh get_random_start_and_direction] Fail to get valid camera")
        return None,None