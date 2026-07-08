import unreal
import math
import time
import json
import os
import random
import datetime
from libs.cameras import save_camera, edit_channel
from navigator.navigator import SceneInfo, SceneTrajectoryInfo, Navigator, TransformationSequence
import libs.collison as lib_collision
import libs.skeleton as lib_skeleton
from typing import List

class Renderer:
    def __init__(self, args, sequence_type='nav'):
        unreal.SystemLibrary.execute_console_command(None, "r.FarClipPlane 10000")  # 100米
        unreal.SystemLibrary.execute_console_command(None, "r.EyeAdaptation.SpeedUp=10.0")  # 100米
        unreal.SystemLibrary.execute_console_command(None, "r.EyeAdaptation.SpeedDown=5.0")  # 100米
        # unreal.SystemLibrary.execute_console_command(None, "sg.AntiAliasingQuality 0")  # 100米
        self.output_path = args.get("output_path", "D:/output")
        self.raw_output_path = args.get("raw_output_path", "D:/output")
        self.sequence_name = args.get("sequence_name", "0")
        self.charactor_config = args.get("charactor", {})
        self.camera_config = args.get("camera", {})
        self.navigator_config = args.get("navigator", {})
        self.collison_config = args.get("collison", {})
        self.start_point_collison_config = args.get("start_point_collison", {})
        self.render_config = args.get("render", {})
        self.width = args.get("width", 1280)
        self.height = args.get("height", 720)
        self.BASE_SHOOT_NUMBER = args.get("BASE_SHOOT_NUMBER", 1)
        
        self.sequence_type = sequence_type
        self.frame_rate = 30
        print("====================Renderer Attributes======================")
        for key in self.__dict__.keys():
            print(key, self.__dict__[key])
        
        world = unreal.EditorLevelLibrary.get_editor_world()
        self.current_level_path = world.get_path_name().split('.')[0] # 去掉后缀 .LevelName
        unreal.log(f"当前关卡路径: {self.current_level_path}")
        self.navigator = Navigator(navigator_config=self.navigator_config, collison_config=self.collison_config)
        self.queue_subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
        self.queue = self.queue_subsystem.get_queue()
        self.queue.delete_all_jobs()  # Clear existing jobs
        self.world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        self.animation_path = args.get('animation_path', None)
    
    def setup_scene(self):
        all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
        for actor in all_actors:
            if isinstance(actor, unreal.CineCameraActor):
                unreal.EditorLevelLibrary.destroy_actor(actor)
                continue
            elif isinstance(actor, unreal.CameraActor):
                unreal.EditorLevelLibrary.destroy_actor(actor)
                continue
            elif isinstance(actor, unreal.CameraRig_Crane):
                unreal.EditorLevelLibrary.destroy_actor(actor)
                continue
            elif isinstance(actor, unreal.SphereReflectionCapture):
                unreal.EditorLevelLibrary.destroy_actor(actor)
                continue
            elif isinstance(actor, unreal.CameraRig_Rail):
                unreal.EditorLevelLibrary.destroy_actor(actor)
                continue
            elif isinstance(actor, unreal.SkeletalMeshActor):
                unreal.EditorLevelLibrary.destroy_actor(actor)
                continue
            elif isinstance(actor, unreal.Character):
                print(f"Destroying Character: {actor.get_name()}")
                unreal.EditorLevelLibrary.destroy_actor(actor)
                continue
            elif isinstance(actor, unreal.TriggerBox):
                unreal.EditorLevelLibrary.destroy_actor(actor)
                continue
            components = actor.get_components_by_class(unreal.SkeletalMeshComponent)
            if components:
                unreal.EditorLevelLibrary.destroy_actor(actor)
                continue
            components = actor.get_components_by_class(unreal.CameraComponent)
            if components:
                unreal.EditorLevelLibrary.destroy_actor(actor)
                continue
        lib_collision.add_box_collision_to_all_meshes()
        self.setup_collison()

    def setup_collison(self):
        # 2. 获取场景中所有 Actor
        all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
        modified_count = 0

        print("[INFO][RENDERER] Starting to set collision profiles to BlockAllDynamic...")

        for actor in all_actors:
            # 查找 StaticMeshComponent (大多数场景物体都基于此)
            # 如果是复杂的组合体，可以使用 get_components_by_class
            mesh_comp = actor.get_component_by_class(unreal.StaticMeshComponent)
            
            if mesh_comp:
                # 设置碰撞使能：同时支持查询（Trace）和物理模拟
                mesh_comp.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS)
                mesh_comp.set_collision_profile_name("BlockAllDynamic")
                modified_count += 1

        print(f"[INFO][RENDERER] Done! Modified {modified_count} actors.")

        unreal.EditorLevelLibrary.save_current_level()
        

    def remove_all_level_sequences(self, asset_path="/Game/Sequences"):
        all_sequences = []
        asset_list = unreal.EditorAssetLibrary.list_assets(asset_path)
        
        for asset_path in asset_list:
            asset = unreal.EditorAssetLibrary.load_asset(asset_path)
            if isinstance(asset, unreal.LevelSequence):
                unreal.log("Level Sequence: " + asset.get_name())
                unreal.EditorAssetLibrary.delete_asset(asset_path)
        return 

    def assign_render_job(self, job, level_sequence, output_path, 
                          enable_dof=False, 
                          render_depth=False,
                          render_motion=False,
                          render_jpg=True,
                          render_exr=False,
                          **kwargs
                          ):
        sequence_path = unreal.EditorAssetLibrary.get_path_name_for_loaded_asset(level_sequence)
        job.sequence = unreal.SoftObjectPath(sequence_path)
        world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        # Set the map to the current level
        asset_path = unreal.EditorAssetLibrary.get_path_name_for_loaded_asset(world)
        job.map = unreal.SoftObjectPath(asset_path)

        # Get configuration
        config = job.get_configuration()

        # Add Deferred Rendering Pass
        deferred_pass = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)


        # 2. 深度通道设置
        if render_depth:
            depth_material_path = '/MovieRenderPipeline/Materials/MovieRenderQueue_WorldDepth.MovieRenderQueue_WorldDepth'
            depth_material = unreal.load_asset(depth_material_path)
            if depth_material:
                deferred_pass_depth = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
                post_process_pass = unreal.MoviePipelinePostProcessPass()
                post_process_pass.material = depth_material
                post_process_pass.enabled = True
                deferred_pass_depth.additional_post_process_materials.append(post_process_pass)
                # 建议给这个 pass 起个名字，方便区分文件
                # deferred_pass_depth.pass_identifier = "Depth"

        # 3. 运动矢量通道设置
        if render_motion:
            motion_material_path = '/MovieRenderPipeline/Materials/MovieRenderQueue_MotionVectors.MovieRenderQueue_MotionVectors'
            motion_material = unreal.load_asset(motion_material_path)
            if motion_material:
                deferred_pass_motion = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
                post_process_pass = unreal.MoviePipelinePostProcessPass()
                post_process_pass.material = motion_material
                post_process_pass.enabled = True
                deferred_pass_motion.additional_post_process_materials.append(post_process_pass)
                # deferred_pass_motion.pass_identifier = "Motion"

        # Add Anti-Aliasing Setting for Warmup (increased to address submission error)
        aa_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
        aa_setting.engine_warm_up_count = 64  # Further increased warmup frames
        aa_setting.render_warm_up_count = 32  # Add some rendered warmup frames
        aa_setting.spatial_sample_count = 2
        aa_setting.temporal_sample_count = 1
        # aa_setting.override_anti_aliasing = True
        # aa_setting.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_TEMPORAL_AA        
        output_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
        output_setting.output_directory = unreal.DirectoryPath(f"{output_path}")
        output_setting.file_name_format = "{sequence_name}.{render_pass}.{frame_number}" # 建议加入 {render_pass} 区分文件名
        output_setting.flush_disk_writes_per_shot = False  # Ensure frames are written properly
        # output_setting.output_resolution = unreal.IntPoint(1920, 1080)  # 4K分辨率，建议从1920x1080开始测试
        output_setting.output_resolution = unreal.IntPoint(self.width, self.height)  # 4K分辨率，建议从1920x1080开始测试

        # --- 核心修改部分：输出格式控制 ---

        # 4. 始终添加 JPG 输出（用于 RGB）
        if render_jpg:
            jpg_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_JPG)

        # 5. 增加 render_exr 选项控制
        # 假设 render_exr 在类初始化中定义
        if render_exr:
            exr_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_EXR)
            # 对于深度图，EXR 通常需要多通道或非压缩模式，视需求而定
            exr_setting.compression = unreal.EXRCompressionFormat.PIZ
        
        cvars_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
        
        # NOTE: fail to import lumen because of square mosaic artifact
        # r.VirtualTextures=False
        # r.Lumen.HardwareRayTracing.LightingMode=2
        # r.Lumen.TraceMeshSDFs=1
        # r.DistanceFields.DefaultVoxelDensity=0.400000
        if kwargs.get("enable_lumen", True):
            base_cvars = {
                'r.VirtualTextures': 0,
                'r.Lumen.HardwareRayTracing.LightingMode':2,
                'r.Lumen.TraceMeshSDFs': 1,
                'r.DistanceFields.DefaultVoxelDensity':0.400000,
                'r.Shadow.Virtual.Enable': 0,           # 关闭 VSM，回归标准阴影贴图 IMPORTANT!!!
            }

        # BUG: without raytracing cause indoor scenes too much dark
        elif kwargs.get("disable_raytracing", False):
            base_cvars = {
                # --- 彻底关闭 Lumen 和实时光追 ---
                'r.DynamicGlobalIlluminationMethod': 0, # 0: None (使用传统光照)
                'r.ReflectionMethod': 0,                # 0: None (使用反射探针)
                'r.RayTracing': 0,                      # 彻底关闭光追开关
                'r.Lumen.DiffuseSetup.DirectLighting': 0,
                
                # --- 强化传统阴影 ---
                'r.Shadow.Virtual.Enable': 0,           # 关闭 VSM，回归标准阴影贴图
                'r.ShadowQuality': 5,                   # 传统阴影最高质量
                'r.Shadow.CSM.MaxCascades': 4,          # 增加级联阴影数量
                'r.Shadow.MaxResolution': 4096,         # 强制阴影贴图为 4K 分辨率
                
                # --- 保持画面清晰 ---
                'r.ScreenPercentage': 100,
                'r.TextureStreaming': 0,
            }
        # BUG: without lumen cause indoor scenes too much dark
        else:
            base_cvars = {
                # --- 基础质量 ---
                'r.ScreenPercentage': 100,
                'r.TextureStreaming': 0,
                'r.Streaming.PoolSize': -1,

                # --- 关闭 Lumen，保留硬件光追 ---
                # 1 表示 Lumen，0 表示 None（传统模式），2 表示 SSGI
                'r.DynamicGlobalIlluminationMethod': 0, 
                # 1 表示 Lumen，2 表示标准光线追踪 (Ray Tracing)
                'r.ReflectionMethod': 2,                

                # --- 硬件光线追踪关键参数 ---
                'r.RayTracing': 1,
                'r.HardwareRayTracing': 1,
                'r.RayTracing.Shadows': 1,              # 启用光追阴影
                'r.RayTracing.Reflections.ScreenSpace': 0, # 禁用屏幕空间反射，强制使用真光追
                'r.RayTracing.Reflections.SamplesPerPixel': 4, # 增加光追反射采样
                'r.RayTracing.Reflections.MaxBounces': 2,

                # --- 解决阴影问题 ---
                # 如果你之前用 VSM 还是有块，建议在此模式下尝试关闭 VSM，使用光追阴影
                'r.Shadow.Virtual.Enable': 1, 
                'r.Shadow.Virtual.ResolutionLodBias': -1,
                
                # --- 维持抗锯齿设置 ---
                'r.TSR.RejectionAntiAliasingQuality': 2,
            }
        # 
        if enable_dof:
            base_cvars.update({'r.DepthOfFieldQuality': 2.0,})
        else:
            base_cvars.update({'r.DepthOfFieldQuality': 0.0,})
        
        for key, value in base_cvars.items():
            cvars_setting.add_or_update_console_variable(key, value)
        for key, value in kwargs.items():
            cvars_setting.add_or_update_console_variable(key, value)

    def execute_async_render(self, job, on_finished_callback):
        """
        通用的异步渲染启动器
        """
        queue = self.subsystem.get_queue()
        self.active_executor = unreal.MoviePipelinePIEExecutor()
        
        # 绑定回调：一旦渲染结束，Unreal 会自动执行 on_finished_callback
        self.active_executor.on_executor_finished_delegate.add_callable(on_finished_callback)
        
        try:
            self.subsystem.render_queue_instance_with_executor_instance(queue, self.active_executor)
        except Exception as e:
            unreal.log_error(f"Failed to start render: {e}")
            self.active_executor = None
    
    def execute_offline(self):
        executor = unreal.MoviePipelineNewProcessExecutor()
        self.queue_subsystem.render_queue_with_executor_instance(executor)
        print("[INFO][RENDERER] Rendering completed.")
    
    def get_direction_light_intensity(self):
        actors = unreal.EditorLevelLibrary.get_all_level_actors()
        directional_lights = [actor for actor in actors if isinstance(actor, unreal.DirectionalLight)]
        for light_actor in directional_lights:
            # 2. 访问光照组件 (LightComponent)
            # 所有的强度、颜色等属性都存储在 LightComponent 中
            light_component = light_actor.light_component
            
            # 3. 获取强度属性
            intensity = light_component.get_editor_property("intensity")
            
            # 打印结果
            actor_name = light_actor.get_actor_label()
            print(f"[INFO][RENDERER] 光源名称: {actor_name} | 强度值: {intensity}")
            return intensity
        print("[INFO][RENDERER] No direction light")
        return 5.0
    
    def initialize_camera(self):
        camera = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).spawn_actor_from_class(unreal.CineCameraActor, unreal.Vector(0,0,10000), unreal.Rotator(0,-45,0))
        # Adjust FOV for wider vertical view to ensure full body inclusion
        camera_comp = camera.get_cine_camera_component()
        camera_comp.set_filmback_preset_by_name("16:9 DSLR")
        camera_comp.current_focal_length = self.camera_config.get("lens", 18)
        camera_comp.current_aperture = 1.2
        camera_comp.post_process_settings.override_motion_blur_amount = True
        camera_comp.post_process_settings.motion_blur_amount = 0.0
        camera_comp.focus_settings.focus_method = unreal.CameraFocusMethod.DISABLE  # 完全禁用DOF
        auto_exposure = self.camera_config.get("auto_exposure", True)
        exposure_bias = self.camera_config.get("exposure_bias", True)
        if auto_exposure:
            # 1. 开启曝光方法的重写 (Override)
            camera_comp.post_process_settings.override_auto_exposure_method = True
            # 2. 将曝光方法设置为 MANUAL (手动/恒定)
            # 这会完全禁用自动适应亮度的功能
            camera_comp.post_process_settings.auto_exposure_method = unreal.AutoExposureMethod.AEM_HISTOGRAM

            camera_comp.post_process_settings.override_auto_exposure_min_brightness = True
            camera_comp.post_process_settings.override_auto_exposure_max_brightness = True
            camera_comp.post_process_settings.auto_exposure_bias = exposure_bias
            print("[INFO][RENDERER] exposure_bias" , camera_comp.post_process_settings.auto_exposure_bias)
            camera_comp.post_process_settings.override_auto_exposure_bias = True
            camera_comp.post_process_settings.auto_exposure_speed_down = 20
            camera_comp.post_process_settings.auto_exposure_speed_up = 20
            camera_comp.post_process_settings.override_auto_exposure_speed_up = True
            camera_comp.post_process_settings.override_auto_exposure_speed_down = True
        else:
            # 1. 开启曝光方法的重写 (Override)
            camera_comp.post_process_settings.override_auto_exposure_method = True
            # 2. 将曝光方法设置为 MANUAL (手动/恒定)
            # 这会完全禁用自动适应亮度的功能
            camera_comp.post_process_settings.auto_exposure_method = unreal.AutoExposureMethod.AEM_MANUAL
            # 3. 设置恒定的曝光补偿值
            # 在 Manual 模式下，Bias 通常直接控制最终画面的 EV (曝光值)
            # print("Fixed Exposure Bias:", exposure_bias)
            # camera_comp.post_process_settings.override_auto_exposure_bias = True
            # camera_comp.post_process_settings.auto_exposure_bias = exposure_bias
            # 注意：之前的 Speed Up/Down 和 Min/Max Brightness 在 Manual 模式下不再需要，已移除。
        # === 裁剪设置 ===
        camera_comp.override_custom_near_clipping_plane = True
        camera_comp.custom_near_clipping_plane = 0.1
        return camera

    def load_skeletal_character(self, character_info):
        skeletal_mesh_path = character_info['skeletal_mesh_path']
        # Spawn the central character actor as SkeletalMeshActor for SK_Sarah
        character_location = unreal.Vector(0,0,0)
        character_rotation_vec = unreal.Vector(0,0,0)
        character_class = unreal.SkeletalMeshActor
        character_rotation = unreal.Rotator(character_rotation_vec.x, character_rotation_vec.y, character_rotation_vec.z)  # Adjust rotation as needed
        character = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).spawn_actor_from_class(character_class, character_location, character_rotation)
        character.set_actor_scale3d(unreal.Vector(1.0, 1.0, 1.0))
        
        # Load and set the skeletal mesh (adjust path to your SK_Sarah mesh)
        skeletal_mesh = unreal.load_asset(skeletal_mesh_path, unreal.SkeletalMesh)
        
        skeletal_mesh_comp = character.get_component_by_class(unreal.SkeletalMeshComponent)
        if skeletal_mesh_comp:
            skeletal_mesh_comp.set_skeletal_mesh(skeletal_mesh)
            # Scale the character to half size
            skeletal_mesh_comp.set_relative_scale3d(unreal.Vector(1, 1, 1))
        else:
            print("[INFO][RENDERER] No skeletal_mesh_comp")
            return -1
        bounds = skeletal_mesh.get_bounds()
        box_extent = bounds.box_extent
        length = box_extent.x * 2.0
        width = box_extent.y * 2.0
        height = box_extent.z * 2.0
        unreal.log(f"[INFO][RENDERER] Skeletal Mesh: {skeletal_mesh.get_name()}")
        unreal.log(f"[INFO][RENDERER] Length (X): {length:.2f} cm")
        unreal.log(f"[INFO][RENDERER] Width (Y): {width:.2f} cm")
        unreal.log(f"[INFO][RENDERER] Height (Z): {height:.2f} cm")
        character_info.update({
            'length':length,
            'width':width,
            'height':height
        })

        props = lib_skeleton.estimate_skeletalmesh_property(skeletal_mesh)
        if props is not None:
            character_info.update(props)
        return character, skeletal_mesh_comp, character_info
    
    def load_bp_character(self, character_info):
        blueprint_path = character_info['blueprint_path']
        character_location = unreal.Vector(0,0,0)
        character_rotation_vec = unreal.Vector(0,0,0)
        character_class = unreal.EditorAssetLibrary.load_blueprint_class(blueprint_path)
        character_rotation = unreal.Rotator(character_rotation_vec.x, character_rotation_vec.y, character_rotation_vec.z)  # Adjust rotation as needed
        character = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).spawn_actor_from_class(character_class, character_location, character_rotation)
        character.set_actor_scale3d(unreal.Vector(1.0, 1.0, 1.0))
    
        skeletal_mesh_comp = None
        components = character.get_components_by_class(unreal.SkeletalMeshComponent)
        if components:
            # 通常第一个组件是主要的骨骼网格体组件
            skeletal_mesh_comp = components[0]
        else:
            # 如果蓝图没有骨骼网格体组件，尝试添加一个
            skeletal_mesh_comp = unreal.SkeletalMeshComponent(character)
            character.add_instance_component(skeletal_mesh_comp)
            skeletal_mesh_comp.register_component()
        # skeletal_mesh_comp.animation_data.anim_to_play = animation_asset
        skeletal_mesh_comp.set_animation_mode(unreal.AnimationMode.ANIMATION_SINGLE_NODE)
        
        skeletal_mesh = skeletal_mesh_comp.skeletal_mesh 
        
        # ADD
        movement_comp = character.get_component_by_class(unreal.CharacterMovementComponent)
        if movement_comp:
            # 关闭重力 (方法1：直接设置重力缩放为0)
            movement_comp.set_editor_property("gravity_scale", 0.0)
        
        bounds = skeletal_mesh.get_bounds()
        box_extent = bounds.box_extent
        length = box_extent.x * 2.0
        width = box_extent.y * 2.0
        height = box_extent.z * 2.0
        unreal.log(f"Skeletal Mesh: {skeletal_mesh.get_name()}")
        unreal.log(f"Length (X): {length:.2f} cm")
        unreal.log(f"Width (Y): {width:.2f} cm")
        unreal.log(f"Height (Z): {height:.2f} cm")
        character_info.update({
            'length':length,
            'width':width,
            'height':height
        })

        bp_info = lib_skeleton.estimate_blueprint_property(unreal.load_asset(blueprint_path))
        character_info.update(bp_info)
        character_info.update({
            'height':bp_info['half_height'] * 2
        })
        # character.set_actor_location(character_location + unreal.Vector(0,0,height/2), sweep=False, teleport=True)
        return character, skeletal_mesh_comp, character_info
    
    def load_character(self, character_info):
        if 'blueprint_path' in character_info:
            return self.load_bp_character(character_info)
        else:
            return self.load_skeletal_character(character_info)
        
    def load_camera_to_sequence(self, camera: unreal.CineCameraActor, level_sequence):
        # Add the camera as a possessable to the sequence
        camera_binding = level_sequence.add_possessable(camera)
        return camera_binding
    
    def set_camera_cut(self, camera_binding, level_sequence, start_frame, end_frame):
        # Set the camera binding ID
        camera_binding_id = unreal.MovieSceneObjectBindingID()
        camera_binding_id.set_editor_property('Guid', camera_binding.get_id())

        # Add a camera cut track (master track)
        camera_cut_track = level_sequence.add_track(unreal.MovieSceneCameraCutTrack)
        camera_cut_section = camera_cut_track.add_section()
        camera_cut_section.set_range(0, end_frame)
        camera_cut_section.set_editor_property('CameraBindingID', camera_binding_id)
    
    def set_transform_track(self, camera: unreal.CineCameraActor, camera_binding, scene_info:SceneInfo):
        # Add a transform track to the camera binding
        transform_track = camera_binding.add_track(unreal.MovieScene3DTransformTrack)
        for traj_info in scene_info.trajectories:
            # Add a section to the transform track
            traj_info:SceneTrajectoryInfo
            if traj_info.extra_info.get('relative', False):
                continue
            print("[INFO][RENDERER] Setting absolute transform for frames: ", traj_info.scene_output_path)
            transform_section = transform_track.add_section()
            transform_section.set_editor_property('use_quaternion_interpolation', True)
            transform_section.set_range(traj_info.start_frame, traj_info.end_frame - 1)
            
            num_keys = len(traj_info.camera.key_frames)
            # Get the channels for location, rotation, and scale
            channels = transform_section.get_all_channels()
            # channels[0-2]: location X, Y, Z
            # channels[3-5]: rotation Roll, Pitch, Yaw
            # channels[6-8]: scale X, Y, Z
            # Add keyframes for the orbit starting from initial angle
            for i in range(num_keys):
                edit_channel(channels, traj_info.start_frame + traj_info.camera.key_frames[i], traj_info.camera.key_frame_locations[i], traj_info.camera.key_frame_rotations[i], 1.0)
            self.save_camera_to_json(camera, traj_info, channels)
    
    def interpolate_times(self, key_frames, key_values, query_idx):
        results = []
        n = len(key_frames)
        for q in query_idx:
            if q <= key_frames[0]:
                results.append(key_values[0])
                continue
            if q >= key_frames[-1]:
                results.append(key_values[-1])
                continue
            lo, hi = 0, n - 1
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if key_frames[mid] <= q:
                    lo = mid
                else:
                    hi = mid
            f0, f1 = key_frames[lo], key_frames[hi]
            v0, v1 = key_values[lo], key_values[hi]
            if f1 == f0:
                results.append(v0)
            else:
                t = (q - f0) / (f1 - f0)
                results.append(v0 + (v1 - v0) * t)
        return results
    
    def save_camera_to_json(self, camera: unreal.CineCameraActor, traj_info:SceneTrajectoryInfo, channels, name="camera_params.json"):
        
        camera_comp = camera.get_cine_camera_component()
        # get camera intrinsics
        hfov_deg = 2 * math.atan(camera_comp.filmback.sensor_width / 2 / camera_comp.current_focal_length)
        hfov_deg = math.degrees(hfov_deg)
        data = save_camera([i for i in range(traj_info.start_frame + self.navigator_config['warmup_frame_num'], traj_info.end_frame)], hfov_deg, channels, width=self.width, height=self.height)
        data['sequence_name'] = self.sequence_name
        if 'scene_frames' in traj_info.extra_info:
            data['times'] = self.interpolate_times(traj_info.camera.key_frames, traj_info.extra_info['scene_frames'], [i for i in range(self.navigator_config['warmup_frame_num'], traj_info.length)])
        os.makedirs(traj_info.scene_output_path, exist_ok=True)
        json_path = os.path.join(traj_info.scene_output_path, name)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def set_camera_focus(self, camera: unreal.CineCameraActor):
        # --- 1. 设置相机的对焦设置 (Focus Settings) ---
        camera_component = camera.get_cine_camera_component()
        focus_settings = camera_component.focus_settings
        camera_component.current_aperture = 1.2
        # 设置对焦方法为追踪 (Tracking)
        focus_settings.focus_method = unreal.CameraFocusMethod.TRACKING
        
        # 设置追踪目标为 scene_info.actor
        # focus_settings.tracking_focus_settings.actor_to_track = scene_info.actor
        
        # 设置对焦平滑插值速度
        focus_settings.focus_smoothing_interp_speed = 1000.0
        
        # 将修改后的设置应用回相机组件
        camera_component.set_editor_property('focus_settings', focus_settings)

        return
    
    def set_focus_actor_track(self, level_sequence, camera: unreal.CineCameraActor, camera_binding, scene_info_list:list[SceneInfo], actor_list:List[unreal.Actor]):
        # --- 4. 关键点：追踪目标 (Actor to Track) 轨道 ---
        # 使用 ObjectPropertyTrack 来切换追踪的对象
        camera_component = camera.get_cine_camera_component()
        camera_component_binding = level_sequence.add_possessable(camera_component)
        camera_component_binding.set_parent(camera_binding)
        actor_track = camera_component_binding.add_track(unreal.MovieSceneObjectPropertyTrack)

        # 路径指向 CineCameraComponent 中的结构体深层属性
        actor_track.set_property_name_and_path('ActorToTrack', 'FocusSettings.TrackingFocusSettings.ActorToTrack')
        # actor_track.set_property_name_and_path('ActorToTrack', 'LookatTrackingSettings.ActorToTrack')
        actor_track.set_object_property_class(unreal.Actor)
        for scene_info,actor in zip(scene_info_list,actor_list):
            actor_section = actor_track.add_section()
            actor_section.set_range(scene_info.start_frame, scene_info.end_frame)
            
            # 获取 Object 通道
            actor_channel = actor_section.get_all_channels()[0]
            # 在起始帧打入关键帧，目标为 scene_info.actor
            # 如果你有多个 actor 随时间切换，可以在循环中多次调用 add_key
            actor_channel.add_key(
                time=unreal.FrameNumber(scene_info.start_frame), 
                new_value=actor
            )
            actor_channel.add_key(
                time=unreal.FrameNumber(scene_info.end_frame-1), 
                new_value=actor
            )
    
    def load_character_to_sequence(self, character, skeletal_mesh_comp, character_info, level_sequence, scene_info:SceneInfo, static=False):
        """
        frames_info = {
            'start_frame': period*cycle_frames,
            'length': [],
            'location': ,
            'rotation': 
        }
        """
        # Add the character as a possessable to the sequence
        character_binding = level_sequence.add_possessable(character)

        transform_track = character_binding.add_track(unreal.MovieScene3DTransformTrack)
        # ================== Animation Track =================
        # Load the animation asset (replace with your animation sequence path)
        animation_asset_path = character_info['animation_asset_path']['random']
        animation_asset = unreal.load_asset(animation_asset_path, unreal.AnimSequence)
        # Add skeletal animation track to the character binding
        if 'blueprint_path' in character_info:
            skeletal_mesh_binding = level_sequence.add_possessable(skeletal_mesh_comp)
            skeletal_mesh_binding.set_parent(character_binding)
            anim_track = skeletal_mesh_binding.add_track(unreal.MovieSceneSkeletalAnimationTrack)
            # transform_track = skeletal_mesh_binding.add_track(unreal.MovieScene3DTransformTrack)
            z_offset = unreal.Vector(0,0,character_info["foot_height_cm"])
            forward_offset_deg = float(character_info["forward_offset_deg"])
            yaw_deg = -forward_offset_deg

            character_rotation_vec = unreal.Vector(0, 0, yaw_deg)
        else:
            skeletal_mesh_binding = character_binding
            # anim_track = character_binding.add_track(unreal.MovieSceneSkeletalAnimationTrack)
            
            anim_track = character_binding.add_track(unreal.MovieSceneSkeletalAnimationTrack)
            z_offset = unreal.Vector(0,0,0)
            forward_offset_deg = float(character_info["forward_offset_deg"])
            yaw_deg = -forward_offset_deg
            character_rotation_vec = unreal.Vector(0, 0, yaw_deg)
        print("[INFO][Renderer] character_rotation_vec", character_rotation_vec)
        print("[INFO][Renderer] z_offset", z_offset, scene_info.character_transformation.key_frame_locations[0])
        for traj_info in scene_info.trajectories:
            start_frame = traj_info.start_frame
            length = traj_info.length

            # Add a section to the transform track
            transform_section = transform_track.add_section()
            transform_section.set_editor_property('use_quaternion_interpolation', True)
            transform_section.set_range(traj_info.start_frame, traj_info.end_frame - 1)
            
            num_keys = len(scene_info.character_transformation.key_frames)
            # Get the channels for location, rotation, and scale
            channels = transform_section.get_all_channels()
            # channels[0-2]: location X, Y, Z
            # channels[3-5]: rotation Roll, Pitch, Yaw
            # channels[6-8]: scale X, Y, Z
            # Add keyframes for the orbit starting from initial angle
            for i in range(num_keys):
                loc = scene_info.character_transformation.key_frame_locations[i] + z_offset
                rot = scene_info.character_transformation.key_frame_rotations[i] + character_rotation_vec
                edit_channel(channels, traj_info.start_frame + scene_info.character_transformation.key_frames[i], loc, rot, 1.0)
            
            # Add a section to the animation track
            anim_section = anim_track.add_section()
            anim_section.set_range(start_frame, start_frame + length - 1)
            # Create params and set the animation
            params = unreal.MovieSceneSkeletalAnimationParams()
            sequence_length = animation_asset.get_play_length()
            print(f"[INFO][RENDERER] Animation sequence length: {sequence_length} seconds")
            params.animation = animation_asset
            if static:
                params.play_rate = unreal.MovieSceneTimeWarpVariant(0.0)
            anim_section.params = params  # Set the params with animation
        
        # ================== Vanish Track =================
        first_start = scene_info.trajectories[0].start_frame
        last_start = scene_info.trajectories[-1].start_frame
        # add vanish 
        vis_track = character_binding.add_track(unreal.MovieSceneVisibilityTrack)
        vanish_section = vis_track.add_section()
        vanish_section.set_range(0, 100000)
        vanish_channels = vanish_section.get_all_channels()[0]
        vanish_channels.add_key(unreal.FrameNumber(0), False) # 第 0 帧设置为“隐藏”
        vanish_channels.add_key(unreal.FrameNumber(first_start), True) # 起始帧设置为“显示”
        vanish_channels.add_key(unreal.FrameNumber(last_start + length), False) # 结束帧设置为“隐藏”
        return level_sequence, character_binding
    
    def setup_sequence(self, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def render(self):
        self.build_sequence()
        self.apply_render()

    def build_sequence(self):
        # NOTE First build navigation then edit levels or failing
        self.navigator.setup()
        self.setup_scene()
        
        scene_output_path = os.path.join(self.raw_output_path, self.sequence_name)
        os.makedirs(scene_output_path, exist_ok=True)
        
        sequence_asset_path = f"/Game/Sequences/{self.sequence_name}.{self.sequence_name}"
        sequence_asset_list = unreal.EditorAssetLibrary.list_assets("/Game/Sequences")
        if sequence_asset_path in sequence_asset_list:
            unreal.log("Delete Level Sequence: " + sequence_asset_path)
            unreal.EditorAssetLibrary.delete_asset(sequence_asset_path)

        level_sequence = self.setup_sequence()
        asset_path = level_sequence.get_path_name()
        # 使用 EditorAssetLibrary 保存资产
        success = unreal.EditorAssetLibrary.save_asset(asset_path)

        if success:
            is_success = True
            unreal.log(f"Save Level Sequence: {asset_path}")
        else:
            is_success = False
            unreal.log_error(f"Save Level Sequence Fail: {asset_path}")
        unreal.LevelSequenceEditorBlueprintLibrary.open_level_sequence(level_sequence)
        # self.execute_offline()
        unreal.EditorLevelLibrary.save_current_level()
        return is_success

    def apply_render(self):

        scene_output_path = os.path.join(self.raw_output_path, self.sequence_name)
        os.makedirs(scene_output_path, exist_ok=True)
        
        sequence_asset_path = f"/Game/Sequences/{self.sequence_name}.{self.sequence_name}"
        sequence_asset_list = unreal.EditorAssetLibrary.list_assets("/Game/Sequences")
        if sequence_asset_path in sequence_asset_list:
            unreal.log("[SUCCESS] Load Level Sequence: " + sequence_asset_path)
            level_sequence = unreal.EditorAssetLibrary.load_asset(sequence_asset_path)
        else:
            unreal.log("[ERROR] Not Found Level Sequence: " + sequence_asset_path)
        job = self.queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        print(f"[INFO][RENDERER] Start assigning ", flush=True)
        os.makedirs(os.path.dirname(scene_output_path), exist_ok=True)
        self.assign_render_job(job, level_sequence, scene_output_path, **self.render_config)
        print(f"[INFO][RENDERER] Finish assigning ", flush=True)
        unreal.LevelSequenceEditorBlueprintLibrary.open_level_sequence(level_sequence)
        self.execute_offline()

    def debug(self):
        scene_output_path = os.path.join(self.output_path, "raw", self.sequence_name)
        os.makedirs(scene_output_path, exist_ok=True)
        
        sequence_asset_path = f"/Game/Sequences/{self.sequence_name}.{self.sequence_name}"
        sequence_asset_list = unreal.EditorAssetLibrary.list_assets("/Game/Sequences")
        if sequence_asset_path in sequence_asset_list:
            unreal.log("Delete Level Sequence: " + sequence_asset_path)
            unreal.EditorAssetLibrary.delete_asset(sequence_asset_path)

        level_sequence = self.setup_sequence()
        
        job = self.queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        print(f"[INFO][RENDERER] Start assigning ", flush=True)
        os.makedirs(os.path.dirname(scene_output_path), exist_ok=True)
        self.assign_render_job(job, level_sequence,scene_output_path, **self.render_config)
        print(f"[INFO][RENDERER] Finish assigning ", flush=True)

        asset_path = level_sequence.get_path_name()
        # 使用 EditorAssetLibrary 保存资产
        success = unreal.EditorAssetLibrary.save_asset(asset_path)

        if success:
            unreal.log(f"Save Level Sequence: {asset_path}")
        else:
            unreal.log_error(f"Save Level Sequence Fail: {asset_path}")
        unreal.LevelSequenceEditorBlueprintLibrary.open_level_sequence(level_sequence)
        # self.execute_offline()

    # def calculate_key_frames(self, tranformation_list:List[TransformationSequence]):
    #     pass
