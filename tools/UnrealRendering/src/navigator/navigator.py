import unreal
import math
import time
import json
import os
import random
import datetime


import importlib
# 1. 导入模块（注意：reload 只能作用于模块对象，不能作用于类）
import libs.math as lib_math
import libs.collison as lib_collison
import libs.cameras as lib_cameras
import params
# 2. 强制刷新这些模块
importlib.reload(lib_collison)
importlib.reload(lib_cameras)
importlib.reload(lib_math)
importlib.reload(params)
# 3. 从刷新后的模块中重新加载具体的类或变量
from libs.cameras import get_camera_pose

class SceneInfo:
    def __init__(self, start_frame, end_frame):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.trajectories = []
        self.character_transformation:TransformationSequence

    

class TransformationSequence:
    def __init__(self, key_frames, key_frame_locations, key_frame_rotations):
        self.key_frames = key_frames
        self.key_frame_locations = key_frame_locations
        self.key_frame_rotations = key_frame_rotations

    def add_warmup(self, warmup_frames):
        self.key_frames = [0, warmup_frames-1] + [warmup_frames + i for i in self.key_frames]
        self.key_frame_locations = 2 * [self.key_frame_locations[0]] + self.key_frame_locations
        self.key_frame_rotations = 2 * [self.key_frame_rotations[0]] + self.key_frame_rotations
            
        
    def get_character_state(self):
        num_keys = len(self.key_frames)
        # iterate each move
        current_key = "idle"
        start_frame = 0
        states = []
        state_key_frames = []
        for i in range(1, num_keys):
            if (self.key_frame_locations[i]-self.key_frame_locations[i-1]).length() < params.STATIC_DISTANCE:
                key = "idle" 
            elif (self.key_frame_rotations[i]-self.key_frame_rotations[i-1]).length() > 10:
                key = "idle"
            else:
                key ="walk"
            # print(i, num_keys, key)
            if key != current_key or i == num_keys-1:
                # next
                states.append(current_key)
                state_key_frames.append(start_frame)
                start_frame = self.key_frames[i-1]
                current_key = key
        return states, state_key_frames
    
    def get_character_state_idx(self):
        num_keys = len(self.key_frames)
        # iterate each move
        current_key = "idle"
        start_frame_id = 0
        states_map = []
        states = []
        state_key_frame_idxs = []
        for i in range(1, num_keys):
            if (self.key_frame_locations[i]-self.key_frame_locations[i-1]).length() < params.STATIC_DISTANCE:
                key = "idle" 
            elif (self.key_frame_rotations[i]-self.key_frame_rotations[i-1]).length() > 10:
                key = "idle"
            else:
                key ="walk"
            states_map.append(key)

            # print(i, num_keys, key)
            if key != current_key or i == num_keys-1:
                # next
                states.append(current_key)
                state_key_frame_idxs.append(start_frame_id)
                start_frame_id = i-1
                current_key = key
        return states, state_key_frame_idxs

class SceneTrajectoryInfo:
    def __init__(self, start_frame, length, scene_output_path=""):
        self.start_frame = start_frame
        self.end_frame = start_frame + length
        self.length = length
        self.scene_output_path = scene_output_path
        self.camera:TransformationSequence
        self.extra_info = {}

class Sampler: # 替换为你的类名
    
    # 初始化时定义缓存变量
    def __init__(self, func):
        # 缓存上一次的采样参数，用于判断是否需要重置权重
        self._sampler_cache_key = None
        self._sampler_weights = []
        self._grid_dims = (0, 0)
        self.check_location = func

    def get_random_point_in_range(self, origin, box_extent):
        # --- 1. 网格参数配置 ---
        # 网格单元大小（单位：厘米/Unreal单位），可根据场景复杂度调整
        # 单元越小，定位越精准，但计算开销略增
        CELL_SIZE = 100.0 
        
        # 计算网格维度
        grid_cols = max(1, int(box_extent.x * 2 // CELL_SIZE))
        grid_rows = max(1, int(box_extent.y * 2 // CELL_SIZE))
        
        # --- 2. 权重矩阵初始化/重置 ---
        # 使用 origin 和 extent 作为 key，如果场景变化则重置权重
        cache_key = (origin.x, origin.y, box_extent.x, box_extent.y)
        if self._sampler_cache_key != cache_key:
            self._sampler_cache_key = cache_key
            # 初始化权重全为1.0 (概率均等)
            self._sampler_weights = [[1.0 for _ in range(grid_cols)] for _ in range(grid_rows)]
            self._grid_dims = (grid_rows, grid_cols)
            print(f"[DEBUG][Sampler] 初始化新区域权重网格: {grid_rows}x{grid_cols}")

        # --- 3. 开始尝试采样 ---
        for i in range(params.MAX_ATTEMPTS):
            # A. 计算当前所有网格的总权重
            total_weight = sum(sum(row) for row in self._sampler_weights)
            
            if total_weight <= 0:
                # 所有点权重都降为0，说明整个区域基本都不可用
                raise Exception("所有区域权重耗尽，无法找到有效点")

            # B. 加权随机选择一个网格
            # 生成一个 0 ~ total_weight 的随机数
            r_val = random.uniform(0, total_weight)
            
            current_sum = 0
            selected_r, selected_c = 0, 0
            
            # 遍历网格寻找落点
            for r in range(grid_rows):
                for c in range(grid_cols):
                    current_sum += self._sampler_weights[r][c]
                    if current_sum >= r_val:
                        selected_r, selected_c = r, c
                        break
                if current_sum >= r_val:
                    break
            
            # C. 在选中的网格内部进行纯随机取点
            # 计算该网格在World空间中的最小边界
            grid_min_x = origin.x - box_extent.x + selected_c * CELL_SIZE
            grid_min_y = origin.y - box_extent.y + selected_r * CELL_SIZE
            
            # 在网格范围内随机
            rand_x = random.uniform(grid_min_x, grid_min_x + CELL_SIZE)
            rand_y = random.uniform(grid_min_y, grid_min_y + CELL_SIZE)
            
            nav_location = unreal.Vector(rand_x, rand_y, origin.z) # 假设Z轴跟随origin
            
            # --- 4. 可用性检测与反馈 ---
            if self.check_location(nav_location):
                print(f"[SUCCESS][Sampler] Get perfect random point at grid [{selected_r},{selected_c}]: ", nav_location)
                return nav_location
            else:
                # --- 5. 失败惩罚 ---
                # 该点不可用，降低该网格的权重，下次采到该网格的概率降低
                # 乘以惩罚因子 (例如 0.5 或 0.8)
                PENALTY_FACTOR = 0.5
                self._sampler_weights[selected_r][selected_c] *= PENALTY_FACTOR
                
                # 可选：如果权重过低，直接设为0，避免频繁访问“死区”
                if self._sampler_weights[selected_r][selected_c] < 0.01:
                    self._sampler_weights[selected_r][selected_c] = 0.0
                    
                print(f"[FAIL][Sampler] Attempt {i}: Failed at grid [{selected_r},{selected_c}]. Weight reduced to {self._sampler_weights[selected_r][selected_c]:.2f}")

        raise Exception("Cannot find random point for this scene (Max attempts reached)")
    
class Navigator:
    def __init__(self, navigator_config, collison_config):
        
        self.navigator_config = navigator_config
        self.collison_config = collison_config
        self.SAFE_VOLUME_RADIUS = navigator_config["SAFE_VOLUME_RADIUS"]
        self.frame_num = navigator_config["frame_num"]
        self.warmup_frame_num = navigator_config["warmup_frame_num"]
        print("[INFO][Navigator] ",navigator_config)
        print("[INFO][Navigator] ",collison_config)
        self.SCENE_TYPE = 'indoor' if navigator_config['indoor'] else 'outdoor'
        
        
    def setup(self):
        self.get_world_bound()
        info = lib_collison.get_scene_info()
        print("[INFO][Navigator] ",info)
        # if info['invalid_count'] > 10:
        #     raise("Invalid Scene")
        self.info = info
        self.scene_scale = (info["max_bound"] - info["min_bound"])
        self.scene_scale = min(self.scene_scale.x, self.scene_scale.y)
        print("[INFO][Navigator] self.scene_scale =",self.scene_scale)

    def destroy_all_block_volume(self):
        all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
        for actor in all_actors:
            if isinstance(actor, unreal.BlockingVolume):
                unreal.EditorLevelLibrary.destroy_actor(actor)
        print("[INFO][Navigator] All Blocking Volume Destroyed")

    def set_block_volume(self, location, extent):
        block_class = unreal.BlockingVolume
        rotation = unreal.Rotator(0.0, 0.0, 0.0)

        block_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
            block_class, 
            location, 
            rotation
        )
        if block_actor is not None:
            unreal.log("成功放置 Blocking Volume。")

            # ----------------------------------------------------------------------
            # 步骤 5: 设置 Actor 的缩放 (Bounds)
            # ----------------------------------------------------------------------
            # NavMesh Bounds Volume 的范围是通过其 Transform 组件的 Scale 属性来控制的。
            # 假设我们想要一个 X=20000, Y=20000, Z=1000 的大区域 (即 X=20m, Y=20m, Z=1m)
            print("[INFO][Navigator] navmesh location", location)
            print("[INFO][Navigator] navmesh scale_vector", extent) # 135 116
            # 设置根组件的相对缩放
            block_actor.root_component.set_relative_scale3d(extent)
            
            print(f"[INFO][Navigator] Blocking Volume 已放置于 {location}，缩放设置为 {extent}。")
    

    def get_world_bound(self):
        all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
        self.min_bound = self.max_bound = None
        self.all_level_actors = []
        self.all_actors = all_actors
        self.playerstart = None
        plane_actors = []
        road_actors = []
        big_actors = []
        use_light_mass = False
        for actor in all_actors:
            
            if isinstance(actor, unreal.DecalActor):
                continue
            if isinstance(actor, unreal.PointLight):
                continue
            if isinstance(actor,unreal.PlayerStart):
                self.playerstart = actor
                print("[INFO][Navigator] PlayerStart found: ", actor, actor.get_actor_location())
            center, bounds = actor.get_actor_bounds(only_colliding_components=False)
            if isinstance(actor, unreal.LightmassImportanceVolume):
                if bounds.x < 300 and bounds.y < 300:
                    print("[INFO][Navigator] Invalid LightmassImportanceVolume Bound use_light_mass", center, bounds, actor)
                    continue
                else:
                    use_light_mass = True
            if self.min_bound is None:
                self.min_bound = center - bounds #+ unreal.Vector(bounds.x / 2, bounds.y / 2, 0)
                self.max_bound = center + bounds #- unreal.Vector(bounds.x / 2, bounds.y / 2, 0)
            else:
                self.min_bound = self.min_bound.get_min(center - bounds) #+ unreal.Vector(bounds.x / 2, bounds.y / 2, 0)
                self.max_bound = self.max_bound.get_max(center + bounds) #- unreal.Vector(bounds.x / 2, bounds.y / 2, 0)

            if "ceiling" in actor.get_actor_label().lower():
                continue
            if isinstance(actor, unreal.StaticMeshActor) and bounds.x > 100 and bounds.y > 100 and bounds.z < 5:
                plane_actors.append(actor)
            if isinstance(actor, unreal.StaticMeshActor) and bounds.x > 1000 and bounds.y > 1000:
                big_actors.append(actor)
            if isinstance(actor, unreal.Landscape):
                big_actors.append(actor)
        if use_light_mass:
            self.min_bound = None
            self.max_bound = None
        for actor in all_actors:
            if isinstance(actor, unreal.LightmassImportanceVolume):
                
                center, bounds = actor.get_actor_bounds(only_colliding_components=False)
                if bounds.x < 300 and bounds.y < 300:
                    print("[INFO][Navigator] Invalid LightmassImportanceVolume Bound", center, bounds, actor)
                    continue
                else:
                    if self.min_bound is None:
                        self.min_bound = center - bounds #+ unreal.Vector(bounds.x / 2, bounds.y / 2, 0)
                        self.max_bound = center + bounds #- unreal.Vector(bounds.x / 2, bounds.y / 2, 0)
                    else:
                        self.min_bound = self.min_bound.get_min(center - bounds) #+ unreal.Vector(bounds.x / 2, bounds.y / 2, 0)
                        self.max_bound = self.max_bound.get_max(center + bounds) #- unreal.Vector(bounds.x / 2, bounds.y / 2, 0)
                    print("[INFO][Navigator] LightmassImportanceVolume Bound", center, bounds, self.min_bound, self.max_bound)
        
        for actor in all_actors:
            # exclude invalid actor type
            if isinstance(actor, unreal.DecalActor):
                continue
            # find start point actor by name
            for key in ["road", "path", "lane", "street", "stone", "pavement", "ground", "floor"]:
                if self.min_bound is not None:
                    center = actor.root_component.get_world_location()
                    if center.x < self.min_bound.x:
                        continue
                    if center.y < self.min_bound.y:
                        continue
                    if center.z < self.min_bound.z:
                        continue
                    if center.x > self.max_bound.x:
                        continue
                    if center.y > self.max_bound.y:
                        continue
                    if center.z > self.max_bound.z:
                        continue
                # do not use get_name
                if key in actor.get_actor_label().lower():
                    road_actors.append(actor)
                    
        print("[INFO][Navigator] RoadActors",len(road_actors))
        if len(road_actors) > 0:
            self.all_level_actors.append(road_actors)
        print("[INFO][Navigator] PlaneActors",len(plane_actors))
        if len(plane_actors) > 0:
            self.all_level_actors.append(plane_actors)
        print("[INFO][Navigator] BigActors",len(big_actors))
        if len(big_actors) > 0:
            self.all_level_actors.append(big_actors)
        
        self.all_level_actors.append(all_actors)
        # if self.min_bound is None:
        #     raise Exception("No NavMeshBoundsVolume Found")

    def test_ground(self, ground_location:unreal.Vector, test_start_height=200, max_hit_distance=1000,
                    min_height=-200, max_height=200):
        SAFE_VOLUME_RADIUS = 20
        hit_distance = lib_collison.get_hit_location_sphere(ground_location+unreal.Vector(0,0,test_start_height), unreal.Vector(0,0,-1), 
                                                     distance=max_hit_distance,
                                                     SAFE_VOLUME_RADIUS=SAFE_VOLUME_RADIUS,
                                                     trace_complex=True) 
        relative_ground_height = hit_distance - test_start_height + SAFE_VOLUME_RADIUS
        if relative_ground_height > max_height:
            print("[DEBUG][Test Ground] FAIL location=", ground_location, "Too high from ground, ", relative_ground_height)
            return False, relative_ground_height
        elif relative_ground_height < min_height:
            print("[DEBUG][Test Ground] FAIL location=", ground_location, "Too low from ground, ", relative_ground_height)
            return False, relative_ground_height
        print("[DEBUG][Test Ground] OK location=", ground_location, "Valid ground, ", relative_ground_height)
            
        return True, relative_ground_height
    
    def surrounding_ground(self, ground_location:unreal.Vector, radius=50, test_start_height=200, max_hit_distance=1000,
                    min_height=-200, max_height=200):
            # 在圆上均匀采样点
        sample_count = 6
        test_location_list= []
        for segment_idx in range(sample_count):
            # 计算当前线段的起点角度
            start_angle = 2 * math.pi * segment_idx / sample_count
            
            # 计算当前线段的终点角度
            end_angle = 2 * math.pi * (segment_idx + 1) / sample_count
            
            # 计算起点坐标
            start_x = radius * math.cos(start_angle)
            start_y = radius * math.sin(start_angle)
            test_location = ground_location + unreal.Vector(start_x, start_y, 0)
            test_location_list.append(test_location)
        valid_list, height_list = self.test_ground_for_points(test_location_list, test_start_height, max_hit_distance, min_height, max_height)
        if not all(valid_list):
            print("[DEBUG][Surrounding Ground] FAIL location=", ground_location, "Invalid surrounding ground, ", valid_list, height_list)
            return False
        # calculate slope
        slope_list = []
        for i in range(1, len(height_list)):
            slope = (height_list[i] - height_list[i-1]) / (radius if i > 0 else 1)  # Assuming 50m spacing
            slope_list.append(slope < self.navigator_config['max_slope'])
        if not all(slope_list):
            print("[DEBUG][Surrounding Ground] FAIL location=", ground_location, "Too steep surrounding ground, ", slope_list)
            return False
        print("[DEBUG][Surrounding Ground] OK location=", ground_location, "Valid surrounding ground, ", valid_list, height_list)
        return True
    
    def test_ground_for_points(self, ground_location_list, test_start_height=200, max_hit_distance=1000,
                    min_height=-200, max_height=200):
        valid_list = []
        height_list = []
        for ground_location in ground_location_list:
            valid, height = self.test_ground(ground_location, test_start_height=test_start_height,
                                             max_hit_distance=max_hit_distance,
                                             max_height=max_height,min_height=min_height)
            valid_list.append(valid)
            height_list.append(height)
        return valid_list, height_list
    
    # def test_floor(self, ground_location, min_roof_height=400, max_hit_distance=10000):
    #     hit_distance = lib_collison.get_hit_location(ground_location+unreal.Vector(0,0,max_hit_distance), unreal.Vector(0,0,-1), distance=max_hit_distance, SAFE_VOLUME_RADIUS=self.SAFE_VOLUME_RADIUS) 
    #     relative_floor_height = (max_hit_distance - self.SAFE_VOLUME_RADIUS - hit_distance)
    #     if relative_floor_height > min_roof_height:
    #         print("[DEBUG][Test Floor] FAIL location=", ground_location, "Got floor at height, ", relative_floor_height)
    #         return False, relative_floor_height
    #     print("[DEBUG][Test Floor] OK location=", ground_location, "No floor at height, ", relative_floor_height)
    #     return True, relative_floor_height

    def test_floor(self, ground_location, min_roof_height=400, max_hit_distance=10000):
        hit_result, hit_result_reverse = lib_collison.get_hit_location_double(ground_location+unreal.Vector(0,0,50), unreal.Vector(0,0,1),max_hit_distance)
        if hit_result['blocking_hit'] and hit_result_reverse['blocking_hit']:
            print("[DEBUG][Test Floor] OK Exist Valid Floor, location=", ground_location)
        elif not hit_result['blocking_hit'] and not hit_result_reverse['blocking_hit']:
            print("[DEBUG][Test Floor] OK No Floor, location=", ground_location)
        else:
            print("[DEBUG][Test Floor] FAIL Exist Invalid Floor, location=", ground_location)
            return False
        return True
    
    def surrounding_collision(self, ground_location, distance=[0,50], height=[0,180]):
        """
        Note that: height+self.SAFE_VOLUME_RADIUS 
        """
        world = unreal.EditorLevelLibrary.get_editor_world()
        sample_counts = [1, 3, 6]  # 分别采样1, 6, 12个线段
        has_hit = False
        for radius_idx, (radius, height) in enumerate(zip(distance, height)):
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
                    ground_location + unreal.Vector(start_x, start_y, height+self.SAFE_VOLUME_RADIUS),  # 起点
                    ground_location + unreal.Vector(end_x, end_y, height+self.SAFE_VOLUME_RADIUS),      # 终点
                    self.SAFE_VOLUME_RADIUS,
                    unreal.TraceTypeQuery.TRACE_TYPE_QUERY2, # TRACE_TYPE_QUERY1=visibility, TRACE_TYPE_QUERY2: camera
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
            print("[DEBUG][Surrounding Collision] FAIL location", ground_location, "Hit surrounding object at radius, ", distance, radius)
            return True
        print("[DEBUG][Surrounding Collision] OK location", ground_location, "No surrounding collision within distance, ", distance, radius)
        return False

    def forward_collison(self, ground_location:unreal.Vector, rotator:unreal.Rotator, distance_rate=None, distance=1000, height=100):
        forward = rotator.get_forward_vector()
        if distance_rate is not None:
            distance = min(distance, self.navigator_config['distance'] * distance_rate)
        hit_distance = lib_collison.get_hit_location_sphere(ground_location+unreal.Vector(0,0,height), forward, distance=distance+self.SAFE_VOLUME_RADIUS, SAFE_VOLUME_RADIUS=self.SAFE_VOLUME_RADIUS) 
        if hit_distance < distance:
            print("[DEBUG][Forward Collision] FAIL location", ground_location, rotator, "Hit object in forward direction at distance, ", hit_distance, distance)
            return True
        print("[DEBUG][Forward Collision] OK location", ground_location, rotator, "No collision in forward direction within distance, ", hit_distance, distance)
        return False
    
    def backward_collison(self, ground_location:unreal.Vector, rotator:unreal.Rotator , distance_rate=None, distance=None, height=100):
        forward = rotator.get_forward_vector()
        if distance_rate is not None:
            distance = min(distance, self.navigator_config['distance'] * distance_rate)
        hit_distance = lib_collison.get_hit_location_sphere(ground_location+unreal.Vector(0,0,height), -forward, distance=distance+self.SAFE_VOLUME_RADIUS, SAFE_VOLUME_RADIUS=self.SAFE_VOLUME_RADIUS)
        if hit_distance < distance:
            print("[DEBUG][Backward Collision] FAIL location", ground_location, "Hit object in backward direction at distance, ", hit_distance, distance)
            return True
        print("[DEBUG][Backward Collision] OK location", ground_location, "No collision in backward direction within distance, ", hit_distance, distance)
        return False

    def forward_collision_with_width(self, ground_location:unreal.Vector, rotator:unreal.Rotator,
                                     distance=1000, width=300, height=100, sample_count=3):
        """
        检测前方具有宽度的矩形区域是否有碰撞。
        从起点沿宽度方向均匀分布多条平行射线，所有射线方向相同。

        Args:
            ground_location: 地面位置
            rotator: 朝向
            distance: 检测距离
            width: 起点处的检测宽度
            height: 检测高度（从 ground_location 向上偏移）
            sample_count: 平行射线数量（奇数时中心线包含在内）

        Returns:
            bool: True 表示有碰撞，False 表示无碰撞
        """
        world = unreal.EditorLevelLibrary.get_editor_world()
        forward = rotator.get_forward_vector()
        right = rotator.get_right_vector()

        base_location = ground_location + unreal.Vector(0, 0, height)

        for i in range(sample_count):
            # 在 [-width/2, +width/2] 范围内均匀分布起点
            if sample_count == 1:
                lateral_offset = 0.0
            else:
                lateral_offset = -width / 2 + width * i / (sample_count - 1)

            ray_start = base_location + right * lateral_offset
            ray_end = ray_start + forward * distance

            hit_results = unreal.SystemLibrary.sphere_trace_multi(
                world,
                ray_start,
                ray_end,
                self.SAFE_VOLUME_RADIUS,
                unreal.TraceTypeQuery.TRACE_TYPE_QUERY2,
                trace_complex=True,
                actors_to_ignore=[],
                draw_debug_type=params.DEBUG_TACE,
                ignore_self=True
            )

            if hit_results is not None and len(hit_results) > 0:
                print(f"[DEBUG][Width Forward Collision] FAIL - Ray {i} offset={lateral_offset:.0f} hit, location={ground_location}")
                return True

        print(f"[DEBUG][Width Forward Collision] OK - location={ground_location}, distance={distance}, width={width}")
        return False

    def forward_slope(self, ground_location:unreal.Vector, rotator:unreal.Rotator,
                    distance=500, max_slope=0.3, step=50):
        """
        检测前方地面坡度，每 step cm 采样一次地面高度，相邻两点坡度不超过 max_slope。

        Args:
            ground_location: 地面位置（已贴地）
            rotator: 朝向
            distance: 检测距离
            max_slope: 最大允许坡度（高度差 / 水平距离）
            step: 采样间距（cm）

        Returns:
            bool: True 表示坡度过大（检测失败），False 表示坡度正常
        """
        forward = rotator.get_forward_vector()
        forward.z = 0
        forward = forward.normal()

        num_steps = max(1, int(distance / step))
        prev_height = None
        for i in range(1, num_steps + 1):
            sample_location = ground_location + forward * (step * i)
            valid, height = self.test_ground(sample_location)
            if not valid:
                print(f"[DEBUG][Forward Slope] FAIL - Step {i} at {sample_location} cannot find ground")
                return True
            if prev_height is not None:
                slope = abs(height - prev_height) / step
                if slope > max_slope:
                    print(f"[DEBUG][Forward Slope] FAIL - Step {i} slope={slope:.3f} > max={max_slope} at {sample_location}")
                    return True
            prev_height = height

        print(f"[DEBUG][Forward Slope] OK - location={ground_location}, distance={distance}, max_slope={max_slope}")
        return False
    
    def test_surrounding(self, ground_location, test_range=[50, 2000], height=100, sample_number=12, min_sample_number=9):
        # ========================== test surrounding objects ==========================
        world = unreal.EditorLevelLibrary.get_editor_world()
        
        has_hit = 0
        for segment_idx in range(sample_number):
            # 计算当前线段的起点角度
            start_angle = 2 * math.pi * segment_idx / sample_number
            # 计算起点坐标
            start_x = test_range[0] * math.cos(start_angle)
            start_y = test_range[0] * math.sin(start_angle)
            
            # 计算终点坐标
            end_x = params.FAR_DISTANCE * self.scene_scale * math.cos(start_angle)
            end_y = params.FAR_DISTANCE * self.scene_scale * math.sin(start_angle)
            
            # 进行碰撞检测
            hit_results = unreal.SystemLibrary.sphere_trace_multi(
                world,
                ground_location + unreal.Vector(start_x, start_y, height),  # 起点
                ground_location + unreal.Vector(end_x, end_y, height),      # 终点
                self.SAFE_VOLUME_RADIUS,
                unreal.TraceTypeQuery.TRACE_TYPE_QUERY2, # TRACE_TYPE_QUERY1=visibility, TRACE_TYPE_QUERY2: camera
                trace_complex=True,
                actors_to_ignore=[],
                draw_debug_type=params.DEBUG_TACE,
                ignore_self=True
            )
                            
            if hit_results is not None and len(hit_results) > 0:
                has_hit += 1
                continue
        if has_hit < min_sample_number:
            print("[DEBUG][Test Surrounding] FAIL location", ground_location, params.FAR_DISTANCE * self.scene_scale, f"Hit surrounding object at {has_hit}/{sample_number} directions, likely at edge of the map.")
            return False
        print("[DEBUG][Test Surrounding] OK location", ground_location, params.FAR_DISTANCE * self.scene_scale, f"Hit surrounding object at {has_hit}/{sample_number} directions, good for navigation.") 
        return True        

    def check_location(self, nav_location, nav_rotator=None,
                       test_roof=None, test_surrounding=None,
                       test_ground=None, surrounding_collison=None,
                       surrounding_ground=None,
                       forward_collison=None, backward_collison=None,
                       forward_collision_with_width=None,
                       forward_slope=None):
        world = unreal.EditorLevelLibrary.get_editor_world()
        
        # ========================== test ground ==========================
        if test_ground is not None:
            ground_valid, ground_height = self.test_ground(nav_location, **test_ground)
            if not ground_valid:
                return False
            nav_location = nav_location - unreal.Vector(0,0,ground_height)
        # ========================== test roof ==========================
        if test_roof is not None:
            roof_valid = self.test_floor(nav_location, **test_roof)
            if not roof_valid:
                return False

        if test_surrounding is not None:
            if not self.test_surrounding(nav_location, **test_surrounding):
                return False
            
        if surrounding_collison is not None:
            if self.surrounding_collision(nav_location, **surrounding_collison):
                return False
        
        if surrounding_ground is not None:
            if not self.surrounding_ground(nav_location, **surrounding_ground):
                return False

        if forward_collison is not None and nav_rotator is not None:
            if self.forward_collison(nav_location, nav_rotator, **forward_collison):
                return False

        if backward_collison is not None and nav_rotator is not None:
            if self.backward_collison(nav_location, nav_rotator, **backward_collison):
                return False

        if forward_collision_with_width is not None and nav_rotator is not None:
            if self.forward_collision_with_width(nav_location, nav_rotator, **forward_collision_with_width):
                return False

        if forward_slope is not None and nav_rotator is not None:
            if self.forward_slope(nav_location, nav_rotator, **forward_slope):
                return False

        print("[SUCCESS][check_location] Get perfect location: ", nav_location)
        if nav_rotator is not None:
            print("[SUCCESS][check_location] With perfect rotation: ", nav_rotator)
        return True
    
    def test_surrounding_distance(self, location, max_hit_distance=1000, sample_count=12):
        world = unreal.EditorLevelLibrary.get_editor_world()
        # 在圆上均匀采样点
        valid_distance = []
        for segment_idx in range(sample_count):
            # 计算当前线段的起点角度
            start_angle = 2 * math.pi * segment_idx / sample_count

            # 计算起点坐标
            start_x = max_hit_distance * math.cos(start_angle)
            start_y = max_hit_distance * math.sin(start_angle)
            # print(location, lookat_vector, distance)
            hit_result = unreal.SystemLibrary.line_trace_single(
                world,
                location,
                location + unreal.Vector(start_x, start_y, 0),
                unreal.TraceTypeQuery.TRACE_TYPE_QUERY2, # TRACE_TYPE_QUERY1=visibility, TRACE_TYPE_QUERY2: camera
                True,               # b_trace_complex
                [],     # actors_to_ignore
                params.DEBUG_TACE,  # draw_debug_type
                True                # b_ignore_self
            )
            hit_result_ = lib_collison.parse_hit_result(hit_result)
            if hit_result_['blocking_hit']:
                valid_distance.append(hit_result_['distance'])
            else:
                valid_distance.append(max_hit_distance)
        return valid_distance
    
    def is_outside_bound(self, position, min_bound, max_bound):
        if position.x > min_bound.x and position.y > min_bound.y and position.z > min_bound.z and position.x < max_bound.x and position.y < max_bound.y and position.z < max_bound.z:
            return False
        return True


    def get_random_start(self, start_point_collison_config):
        """
        start_point_collison: dict
        """
        i = 0
        while i < params.MAX_ATTEMPTS:
            # use different types of actor as search anchor
            random.seed(time.time())
            actor_level = random.randint(0, len(self.all_level_actors)-1)
            print("[INFO][get_random_start] Trying actor level", actor_level)
            actor = self.all_level_actors[actor_level][random.randint(0, len(self.all_level_actors[actor_level])-1)]
            if actor is None or actor.root_component is None:
                print("[ERROR][get_random_start] Invalid actor", actor)
                continue
            # origin = actor.root_component.get_world_location()
            origin, box_extent = actor.get_actor_bounds(only_colliding_components=False)
            print("[INFO][get_random_start] Use actor", actor.get_actor_label())
            print(f"[INFO][get_random_start] Actor 中心点: {origin}")
            print(f"[INFO][get_random_start] Actor 范围 (Extent): {box_extent}")
            offset = unreal.Vector(random.randint(-int(box_extent.x)-params.SEARCH_SPACE,int(box_extent.x)+params.SEARCH_SPACE), random.randint(-int(box_extent.y)-params.SEARCH_SPACE,int(box_extent.y)+params.SEARCH_SPACE), random.randint(-int(box_extent.z),int(box_extent.z)))
            
            origin = origin + offset
            print("[INFO][get_random_start] Trying random location, ", i, origin, offset)
            if self.is_outside_bound(origin, self.min_bound, self.max_bound):
                print("[FAIL][get_random_start] outside bound, ", i, origin, self.min_bound, self.max_bound)
                continue
            nav_location = origin
            print("[INFO][get_random_start] Random", nav_location)
            i += 1
            ground_valid, ground_height = self.test_ground(nav_location, **start_point_collison_config['test_ground'])
            if not ground_valid:
                continue
            nav_location = nav_location - unreal.Vector(0,0,ground_height)
            if self.check_location(nav_location, nav_rotator=None, **start_point_collison_config):
                print("[SUCCESS][get_random_start] Get perfect camera: ", nav_location)
                return nav_location
        print("[ERROR][get_random_start] Fail to get valid camera")
        return None

    def get_random_start_and_direction(self, start_point_collison_config):
        """
        start_point_collison: dict
        """
        i = 0
        while i < params.MAX_ATTEMPTS:
            random.seed(time.time())
            actor_level = random.randint(0, len(self.all_level_actors)-1)
            print("[INFO][get_random_start_and_direction] Trying actor level", actor_level)
            actor = self.all_level_actors[actor_level][random.randint(0, len(self.all_level_actors[actor_level])-1)]
            if actor is None or actor.root_component is None:
                print("[ERROR][get_random_start_and_direction] Invalid actor", actor)
                continue
            # origin = actor.root_component.get_world_location()
            origin, box_extent = actor.get_actor_bounds(only_colliding_components=False)
            print("[INFO][get_random_start_and_direction] Use actor", actor.get_actor_label())
            print(f"[INFO][get_random_start_and_direction] Actor 中心点: {origin}")
            print(f"[INFO][get_random_start_and_direction] Actor 范围 (Extent): {box_extent}")
            offset = unreal.Vector(random.randint(-int(box_extent.x)-params.SEARCH_SPACE,int(box_extent.x)+params.SEARCH_SPACE), random.randint(-int(box_extent.y)-params.SEARCH_SPACE,int(box_extent.y)+params.SEARCH_SPACE), random.randint(-int(box_extent.z),int(box_extent.z)))
            
            origin = origin + offset
            print("[INFO][get_random_start_and_direction] Trying random location, ", i, origin)
            nav_location = origin
            if self.is_outside_bound(origin, self.min_bound, self.max_bound):
                print("[FAIL][get_random_start] outside bound, ", i, origin, self.min_bound, self.max_bound)
                continue
            i += 1
            print("[INFO][get_random_start_and_direction] Random", nav_location)
            ground_valid, ground_height = self.test_ground(nav_location, **start_point_collison_config['test_ground'])
            if not ground_valid:
                continue
            nav_location = nav_location - unreal.Vector(0,0,ground_height)

            
            if self.check_location(nav_location, nav_rotator=None, **start_point_collison_config):
                print("[SUCCESS][get_random_start_and_direction] Get perfect camera: ", nav_location)
                try:
                    nav_rotator = self.get_valid_rotation(nav_location, unreal.Vector(0,0,0), 180,
                                        start_point_collison_config.get("forward_collison",None),
                                        start_point_collison_config.get("backward_collison",None),
                                        start_point_collison_config.get("forward_collision_with_width",None),
                                        start_point_collison_config.get("forward_slope",None))
                except Exception as e:
                    print("[FAIL][get_random_start_and_direction] Failed to get valid rotation, ", e)
                    continue
                return nav_location, nav_rotator
        print("[ERROR][get_random_start_and_direction] Fail to get valid camera")
        return None,None

    def get_random_point_in_range_sampler(self, origin, box_extent, collison_config):
        check_func = lambda nav_location: self.check_location(nav_location, nav_rotator=None,**collison_config)
        sampler = Sampler(check_func)
        return sampler.get_random_point_in_range(origin, box_extent)

    def get_random_point_in_range(self, origin, box_extent):
        return origin + unreal.Vector(random.randint(int(- box_extent.x), int(box_extent.x)), random.randint(int(- box_extent.y), int(box_extent.y)), 0)

    def validate_camera_pose(self, cam_loc, cam_rot, target_actor, safe_distance=0):
        """
        返回: (is_perfect, is_center_visible, reason)
        """
        world = unreal.EditorLevelLibrary.get_editor_world()
        if target_actor is not None:
            ignore_actors = [target_actor] + list(target_actor.get_attached_actors())
        else:
            ignore_actors = []
        trace_channel = unreal.TraceTypeQuery.TRACE_TYPE_QUERY1 
        cam_rot = lib_math.vec_to_rotator(cam_rot)
        cam_forward = cam_rot.get_forward_vector()
        # === Check A: 相机位置是否就在墙里或者离墙太近？(运动安全性检测) ===
        sphere_hit_result = unreal.SystemLibrary.sphere_trace_single(
            world,
            cam_loc, # st loc
            cam_loc + cam_forward * safe_distance, # ed loc
            self.SAFE_VOLUME_RADIUS,  # 半径
            trace_channel, # 通常 Foliage 是 WorldStatic，Query1 应该能盖住
            True,                # bTraceComplex=True (关键！开启复杂碰撞检测，能扫到叶片)
            ignore_actors,       # 忽略列表
            params.DEBUG_TACE, # 调试时改成 FOR_DURATION
            True,                # IgnoreSelf
            unreal.LinearColor.RED,
            unreal.LinearColor.GREEN,
            0.0
        )

        # 判定逻辑
        if sphere_hit_result:
            if lib_collison.parse_hit_result(sphere_hit_result)["blocking_hit"]:
                return False, "Camera safety volume collision (Too cluttered)"
        return True, f"Camera location {cam_loc}, {cam_rot} OK"

    # 随机生成相机的目标位置
    def random_position_within_radius(self, center_location, min_radius=0, max_radius=10):
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(min_radius, max_radius)
        x_offset = distance * math.cos(angle)
        y_offset = distance * math.sin(angle)
        z_offset = random.uniform(-1, 1)  # 控制相机在Z轴上下浮动

        return unreal.Vector(center_location.x + x_offset, 
                                center_location.y + y_offset, 
                                center_location.z + z_offset)
        

    def one_step_fixed_lookat(self, num_keys, initial_location, initial_rotation_vector, character_location, check_straight=False):
        for i in range(params.MAX_ATTEMPTS):
            target_rotation = unreal.Vector(0, random.randint(-self.cam_settings['y_rot'],self.cam_settings['y_rot']), random.randint(-self.cam_settings['z_rot'],self.cam_settings['z_rot']))
            target_radius = random.uniform(self.cam_settings['zoom_in'], self.cam_settings['zoom_out'])
            key_frame_locations, key_frame_rotations = get_camera_pose(num_keys, initial_location, initial_rotation_vector, character_location, target_rotation, target_radius)
            hit = False
            last_location = initial_location
            for i in range(num_keys):
                             
            
                candidate_location = key_frame_locations[i]
                if i > 0:
                    last_location = key_frame_locations[i-1]
                target_dis = candidate_location.distance(last_location)
                hit_distance = lib_collison.get_hit_location(last_location, (candidate_location-last_location).normal(), distance=target_dis)
                if hit_distance < target_dis:
                    hit = True
            if hit:
                continue
            else:
                break
        if hit:
            raise Exception("No random rotating traj")
        return key_frame_locations, key_frame_rotations
     
    def get_valid_rotation(self, initial_location_ground, initial_rotation_vector, yaw_range, forward_collison, backward_collison, forward_collision_with_width=None, forward_slope=None):
        """
        基于区域失败概率的旋转角度采样
        Args:
            initial_location_ground: 初始位置
            initial_rotation_vector: 初始旋转向量
            yaw_range: 最大偏航角范围
            forward_collison: forward collision 配置
            backward_collison: backward collision 配置
            forward_collision_with_width: forward collision with width 配置
        """
        # --- 1. 配置参数 ---
        GRID_STEP = 15.0  # 角度网格步长（度）
        PENALTY_FACTOR = 0.5  # 失败惩罚因子

        pitch_range = self.navigator_config.get('y_rot', 0)

        # 计算网格维度 (ceil 保证完整覆盖 [-range, range]，非整除时末尾 bin 会被夹取)
        pitch_bins = max(1, math.ceil(pitch_range * 2 / GRID_STEP)) if pitch_range > 0 else 1
        yaw_bins = max(1, math.ceil(yaw_range * 2 / GRID_STEP)) if yaw_range > 0 else 1

        # --- 2. 初始化权重矩阵 (每次调用都重置；本函数不跨调用缓存) ---
        _rot_sampler_weights = [[1.0 for _ in range(yaw_bins)] for _ in range(pitch_bins)]

        # --- 3. 开始采样循环 ---
        for i in range(params.MAX_ATTEMPTS):
            # A. 展平权重并做加权随机选择，避免手写前缀和的浮点 edge case
            flat_weights = [w for row in _rot_sampler_weights for w in row]
            if sum(flat_weights) <= 0:
                raise Exception("所有旋转角度区域权重耗尽，无法找到有效方向")
            flat_idx = random.choices(range(len(flat_weights)), weights=flat_weights, k=1)[0]
            selected_p_idx, selected_y_idx = divmod(flat_idx, yaw_bins)

            # B. 在选中的网格内随机生成具体角度；末尾 bin 上界夹到 range
            if pitch_range > 0:
                min_pitch = -pitch_range + selected_p_idx * GRID_STEP
                max_pitch = min(min_pitch + GRID_STEP, pitch_range)
                rand_pitch = random.uniform(min_pitch, max_pitch)
            else:
                rand_pitch = 0.0

            if yaw_range > 0:
                min_yaw = -yaw_range + selected_y_idx * GRID_STEP
                max_yaw = min(min_yaw + GRID_STEP, yaw_range)
                rand_yaw = random.uniform(min_yaw, max_yaw)
            else:
                rand_yaw = 0.0

            # C. 构造 Rotator：在初始欧拉角 (x=Roll, y=Pitch, z=Yaw) 上叠加 pitch/yaw 偏移
            target_rotation_rotator = lib_math.vec_to_rotator(
                initial_rotation_vector + unreal.Vector(0, rand_pitch, rand_yaw)
            )

            # --- 4. 碰撞检测与反馈 ---
            if forward_collison is not None and self.forward_collison(initial_location_ground, target_rotation_rotator, **forward_collison):
                _rot_sampler_weights[selected_p_idx][selected_y_idx] *= PENALTY_FACTOR
            elif backward_collison is not None and self.backward_collison(initial_location_ground, target_rotation_rotator, **backward_collison):
                _rot_sampler_weights[selected_p_idx][selected_y_idx] *= PENALTY_FACTOR
            elif forward_collision_with_width is not None and self.forward_collision_with_width(initial_location_ground, target_rotation_rotator, **forward_collision_with_width):
                _rot_sampler_weights[selected_p_idx][selected_y_idx] *= PENALTY_FACTOR
            elif forward_slope is not None and self.forward_slope(initial_location_ground, target_rotation_rotator, **forward_slope):
                _rot_sampler_weights[selected_p_idx][selected_y_idx] *= PENALTY_FACTOR
            else:
                print(f"[SUCCESS][get_valid_rotation] Good turn angle: {initial_location_ground}, {target_rotation_rotator} pitch={rand_pitch} yaw={rand_yaw}")
                return target_rotation_rotator

        raise Exception("Cannot find valid rotation angle (Max attempts reached)")
    
    def get_location_to_ground(self, location):
        valid, height = self.test_ground(location, **self.collison_config.get("test_ground",{}))
        location = location - unreal.Vector(0,0,height)
        return valid, location

    def _slerp_rotator(self, r1_euler, r2_euler, alpha):
        """
        将输入的 Roll, Pitch, Yaw 向量转换为四元数进行球面插值
        r1_euler, r2_euler: unreal.Vector 或类似对象 (x=Roll, y=Pitch, z=Yaw)
        alpha: 插值权重 (0.0 - 1.0)
        """
        # 限制 alpha 范围
        alpha = max(0.0, min(1.0, alpha))
        
        # 1. 将欧拉角向量转换为 Unreal Rotator (顺序: Pitch, Yaw, Roll)
        # 注意：Unreal 的 Rotator 构造函数参数顺序通常是 (P, Y, R)
        rot1 = lib_math.vec_to_rotator(r1_euler)
        rot2 = lib_math.vec_to_rotator(r2_euler)
        res_quat = rot1.quaternion().slerp_quat(rot2.quaternion(), alpha)
        res_rot = res_quat.rotator()
        return lib_math.rotator_to_vec(res_rot)

    def calculate_rotation_angle(self, r1_euler, r2_euler):
        rot1 = lib_math.vec_to_rotator(r1_euler)
        rot2 = lib_math.vec_to_rotator(r2_euler)
        # Note that it is radian not degrees
        angle = rot1.quaternion().angular_distance(rot2.quaternion()) * 180 / math.pi
        return angle
        
    def interpolate_key_frames(self, key_locations, key_rotations, max_distance=25.0, max_rotation=5, decompose_RT=True):
        """
        根据距离对位置进行线性插值，并对旋转进行重复填充
        :param key_locations: list[unreal.Vector]
        :param key_rotations: list[unreal.Vector/unreal.Rotator]
        :param max_distance: 相邻点之间的最大距离
        :return: (new_locations, new_rotations)
        """
        if len(key_locations) != len(key_rotations):
            unreal.log_error("错误：位置列表和旋转列表的长度不一致！")
            return [], []

        new_locations = [key_locations[0]]
        new_rotations = [key_rotations[0]]

        # 遍历到倒数第二个元素
        for i in range(len(key_locations) - 1):
            loc_start = key_locations[i]
            loc_end = key_locations[i+1]
            rot_start = key_rotations[i]
            rot_end = key_rotations[i+1]

            # 计算两个向量之间的距离
            # Unreal Vector 支持减法和 .length()
            diff = loc_end - loc_start
            distance = diff.length()
            rot_diff = self.calculate_rotation_angle(rot_start, rot_end)

            print("[DEBUG][interpolate_key_frames] ", i, loc_start, rot_start, rot_end, rot_diff)
            # 计算需要分成几段 (向上取整)
            num_segments = math.ceil(distance / max_distance)
            
            # 如果距离本来就很小，num_segments 至少为 1
            # num_segments = max(1, num_segments)
            # new_locations.append(loc_start)
            # new_rotations.append(rot_start)
            # new_locations.append(loc_start)
            # new_rotations.append(rot_start)
            if decompose_RT:
                new_locations.append(loc_start)
                new_rotations.append(rot_end) # use a keyframe to let actor rotate
            else:
                num_segments = max(num_segments, math.ceil(abs(rot_diff) / max_rotation))
            # new_rotations.append(rot_start) # no keyframe to let actor rotate, only leave a idle keyframe
            for j in range(1, num_segments+1):
                # 计算插值比例 alpha (0.0 到 1.0)
                alpha = j / float(num_segments)
                
                # 位置插值: start + (end - start) * alpha
                # interp_loc = unreal.MathLibrary.lerp(loc_start, loc_end, alpha)
                interp_loc = loc_start + (diff * alpha)
                new_locations.append(interp_loc)
                if decompose_RT:
                    # 旋转不做插值，直接重复当前段的最后旋转
                    new_rotations.append(rot_end)
                else:
                    new_rotations.append(self._slerp_rotator(rot_start, rot_end, alpha))

        return new_locations, new_rotations

    def allocate_key_frames(self, key_frame_locations, key_frame_rotations):
        key_num = len(key_frame_locations)
        key_frames = list(range(0, key_num))
        if len(key_frames) > 1 and key_frames[-1] > 0:
            target_last_frame = self.frame_num - 1
            actual_last_frame = key_frames[-1]
            
            # 计算缩放系数
            scale_factor = target_last_frame / actual_last_frame
            
            # 重新映射每一帧，并取整
            # 注意：为了保证最后一帧绝对精准，我们单独处理最后一个元素
            scaled_key_frames = []
            for f in key_frames[:-1]:
                scaled_key_frames.append(round(f * scale_factor))
            
            scaled_key_frames.append(target_last_frame)
            key_frames = scaled_key_frames
        print("[DEBUG][RandomWalk Navigate] allocate_key_frames key_num", key_num, len(key_frames), key_frames)
        # 打印调试信息
        print(f"[DEBUG][RandomWalk Navigate] Key Points: {key_num}, Generated Frames: {len(key_frames)}, Frames: {key_frames}")
        return key_frames


class PotentialFieldSampler(Sampler):
    def __init__(self, func):
        super().__init__(func)
        self._sampler_cache_key = None

    def get_directed_random_point(self, origin, box_extent, ideal_direction: unreal.Vector, focus_weight=3.0):
        """
        ideal_direction: V_ideal
        focus_weight: 导向强度
        """
        CELL_SIZE = 100.0 
        
        grid_cols = max(1, int(box_extent.x * 2 // CELL_SIZE))
        grid_rows = max(1, int(box_extent.y * 2 // CELL_SIZE))
        
        # 将 ideal_direction 粗略离散化，用于缓存 Key，避免浮点抖动导致频繁刷新
        dir_key = (round(ideal_direction.x, 1), round(ideal_direction.y, 1))
        cache_key = (origin.x, origin.y, box_extent.x, box_extent.y, dir_key)
        
        if self._sampler_cache_key != cache_key:
            self._sampler_cache_key = cache_key
            self._sampler_weights = [[0.0 for _ in range(grid_cols)] for _ in range(grid_rows)]
            self._grid_dims = (grid_rows, grid_cols)
            
            for r in range(grid_rows):
                for c in range(grid_cols):
                    grid_center_x = origin.x - box_extent.x + c * CELL_SIZE + CELL_SIZE * 0.5
                    grid_center_y = origin.y - box_extent.y + r * CELL_SIZE + CELL_SIZE * 0.5
                    grid_center = unreal.Vector(grid_center_x, grid_center_y, origin.z)
                    
                    vec_to_grid = grid_center - origin
                    
                    if vec_to_grid.length() < 1.0:
                        self._sampler_weights[r][c] = 1.0
                        continue
                        
                    dir_to_grid = vec_to_grid.normal()
                    
                    dot_product = dir_to_grid.x * ideal_direction.x + dir_to_grid.y * ideal_direction.y
                    
                    weight = math.exp(focus_weight * dot_product)
                    
                    self._sampler_weights[r][c] = max(0.01, weight)
                    
            print(f"[PotentialSampler] 场权重已刷新。网格大小: {grid_rows}x{grid_cols}")

        for i in range(params.MAX_ATTEMPTS):
            total_weight = sum(sum(row) for row in self._sampler_weights)
            if total_weight <= 0:
                raise Exception("所有区域权重耗尽，场采样失败")

            r_val = random.uniform(0, total_weight)
            current_sum = 0
            selected_r, selected_c = 0, 0
            
            for r in range(grid_rows):
                for c in range(grid_cols):
                    current_sum += self._sampler_weights[r][c]
                    if current_sum >= r_val:
                        selected_r, selected_c = r, c
                        break
                if current_sum >= r_val:
                    break
            
            grid_min_x = origin.x - box_extent.x + selected_c * CELL_SIZE
            grid_min_y = origin.y - box_extent.y + selected_r * CELL_SIZE
            rand_x = random.uniform(grid_min_x, grid_min_x + CELL_SIZE)
            rand_y = random.uniform(grid_min_y, grid_min_y + CELL_SIZE)
            
            nav_location = unreal.Vector(rand_x, rand_y, origin.z)
            
            if self.check_location(nav_location):
                return nav_location
            else:
                PENALTY_FACTOR = 0.2 
                self._sampler_weights[selected_r][selected_c] *= PENALTY_FACTOR
                if self._sampler_weights[selected_r][selected_c] < 0.001:
                    self._sampler_weights[selected_r][selected_c] = 0.0
                    
        raise Exception("Cannot find valid potential point (Max attempts reached)")
