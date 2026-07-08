import unreal
import math
import time
import json
import os
import random
import datetime
from params import *
from renderer.renderer import Renderer
from navigator.orbit_camera import OrbitCamera
from navigator.navigator import SceneInfo, SceneTrajectoryInfo, TransformationSequence
import libs.math as lib_math

class Renderer_V2V(Renderer):
    """ Renderer for video 2 video camera control
    generate multi-camera synchronized videos

    Args:
        Renderer (_type_): _description_
    """
    def __init__(self, args):
        super().__init__(args, sequence_type="v2v")
        self.navigator = OrbitCamera(self.navigator_config, self.collison_config)
        character_path = args.get("character_path", "info/character.json")
        self.camera_type = self.navigator_config.get("camera_type", ["center"])
    
        with open(character_path, "r") as f:
            character_list = json.load(f)
        with open(self.animation_path, "r") as f:
            animation_list = json.load(f)
        self.character_list = []
        while len(self.character_list) < self.BASE_SHOOT_NUMBER:
            character = character_list[random.randint(0,len(character_list)-1)]
            if character.get("animation_asset_path", None) is None:
                continue
            if "random" not in character.get("animation_asset_path"):
                continue
            character["animation_asset_path"]["random"] = animation_list[random.randint(0,len(animation_list)-1)]
            self.character_list.append(character)
        print("[INFO][V2V] character_list", self.character_list)
     
    def arange_frames(self, character_info, start_frame):
        # =================== Get Random Location from predefined list ===================
        cam_distance = random.randint(self.navigator_config['min_distance'], self.navigator_config['max_distance'])
        # cam_distance = 500
        cam_height = 100
        charactor_height = character_info['height']
        print("[INFO][V2V] Start search camera location: ")
        random_location, initial_rotation_rot = self.navigator.get_random_start_and_direction(self.start_point_collison_config)
        # Add pitch
        if random_location is None:
            print("[ERROR][V2V] cannot find valid Camera & character location")
            return None
        rand_pitch = random.randint(-5,5)
        character_rot = unreal.Rotator(initial_rotation_rot.roll,initial_rotation_rot.pitch + rand_pitch, initial_rotation_rot.yaw)
        
        character_location = random_location
        initial_location = character_rot.get_forward_vector() * cam_distance + character_location + unreal.Vector(0,0,charactor_height*3 / 4)
        

        # 1. 确保向量是归一化的 (Vector b)
        forward = - character_rot.get_forward_vector().normal()
        
        # 2. 计算右向量 (Y 轴)
        # 通过 [0,0,1] 与 目标法线 叉乘，得到一个处于水平面内的右向量
        # 这步是保证 Roll 为 0 的关键
        world_up = unreal.Vector(0.0, 0.0, 1.0)
        cam_rotator = unreal.MathLibrary.make_rot_from_xz(forward, world_up)
        initial_cam_rot_vector = unreal.Vector(cam_rotator.roll, cam_rotator.pitch, cam_rotator.yaw)
        print("[INFO][V2V] character_location cam_location",cam_distance, character_location, initial_rotation_rot, initial_location, cam_rotator)        

        if 'blueprint_path' in character_info:
            character_info['name'] = character_info['name'] + "_" + os.path.basename(character_info['blueprint_path'])
        else:
            character_info['name'] = character_info['name'] + "_" + os.path.basename(character_info['skeletal_mesh_path'])
        character_rotation_vec = lib_math.rotator_to_vec(character_rot)
        print("[INFO][V2V] final charactor: ", character_info, character_location, character_rotation_vec)
        trajectories = []
        current_frame = start_frame
        for cam_name in self.camera_type:
            print("[INFO][V2V] Arranging frames for camera:", cam_name)
            try:
                traj_info:SceneTrajectoryInfo = self.navigator.get_cameras(cam_name, None, initial_location, initial_cam_rot_vector, cam_distance, current_frame)
            except Exception as e:
                print(f"[FAIL][V2V] Error occurred while getting camera info for {cam_name}: {e}")
                continue
            traj_info.scene_output_path =  f"{self.output_path}/{self.sequence_name}/{character_info['name']}/{cam_name}"
            current_frame = traj_info.end_frame
            trajectories.append(traj_info)
        scene_info = SceneInfo(start_frame, current_frame)
        scene_info.character_transformation = TransformationSequence(
            key_frames=[0],
            key_frame_locations=[character_location], 
            key_frame_rotations=[character_rotation_vec]
        )
        scene_info.trajectories = trajectories
        return scene_info
     
    def setup_sequence(self, **kwargs):
        character_list = self.character_list
        camera = self.initialize_camera()
        # Create a new Level Sequence asset
        level_sequence = unreal.AssetToolsHelpers.get_asset_tools().create_asset(self.sequence_name, "/Game/Sequences", unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
        # Set the display rate
        level_sequence.set_display_rate(unreal.FrameRate(self.frame_rate, 1))
        
        current_frame = 0
        scene_info_list = []
        character_actor_list = []
        valid_character_list = []
        text_label = f"Working on {self.sequence_name}"
        with unreal.ScopedSlowTask(len(character_list), text_label) as slow_task:
            slow_task.make_dialog(True)               # Makes the dialog visible, if it isn't already
            for i,character_info in enumerate(character_list):
                if slow_task.should_cancel():         # True if the user has pressed Cancel in the UI
                    exit(0)
                slow_task.enter_progress_frame(1)
                
                character_info['name'] = f"scene_{i:03d}"
                character, skeletal_mesh_comp, character_info  = self.load_character(character_info)
                
                print("[INFO][V2V] Arranging frames for character:", character_info['name'])
                scene_info = self.arange_frames(character_info, start_frame=current_frame)
                if scene_info is None:
                    print("[ERROR] Arranging Frame Fails")
                    unreal.EditorLevelLibrary.destroy_actor(character)
                    continue
                if len(scene_info.trajectories) == 0:
                    print(f"[ERROR][V2V] No valid trajectory for character {character_info['name']}, skipping...")
                    unreal.EditorLevelLibrary.destroy_actor(character)
                    continue
                current_frame = scene_info.end_frame
                scene_info_list.append(scene_info)
                character_actor_list.append((character,skeletal_mesh_comp))
                valid_character_list.append(character_info)
        # Set the playback range for the sequence
        level_sequence.set_playback_start(0)
        level_sequence.set_playback_end(current_frame)
        camera_binding = self.load_camera_to_sequence(camera, level_sequence)
        self.set_camera_cut(camera_binding, level_sequence, 0, current_frame)
        if self.render_config.get("enable_dof", False):
            self.set_camera_focus(camera)
        actor_list = []
        text_label = f"Working on {self.sequence_name}"
        with unreal.ScopedSlowTask(len(valid_character_list), text_label) as slow_task:
            slow_task.make_dialog(True)               # Makes the dialog visible, if it isn't already
            for scene_info, character_info, (character_actor,skeletal_mesh_comp) in zip(scene_info_list, valid_character_list, character_actor_list):
                if slow_task.should_cancel():         # True if the user has pressed Cancel in the UI
                    exit(0)
                slow_task.enter_progress_frame(1)
                
                # scene_info
                self.set_transform_track(camera, camera_binding, scene_info)
                # # set charactor location
                # character_actor.set_actor_location_and_rotation(scene_info.location, unreal.Rotator(scene_info.rotation.x, scene_info.rotation.y, scene_info.rotation.z), sweep=False, teleport=True)
        
                self.load_character_to_sequence(character_actor, skeletal_mesh_comp, character_info, level_sequence, scene_info, **self.charactor_config)
                
                actor_list.append(character_actor)
            if self.render_config.get("enable_dof", False):
                print("[INFO][V2V] Set focus track")
                self.set_focus_actor_track(level_sequence, camera, camera_binding, scene_info_list, actor_list)
        return level_sequence
