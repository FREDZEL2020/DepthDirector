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
# 2. 强制刷新这些模块
importlib.reload(lib_math)
# 3. 从刷新后的模块中重新加载具体的类或变量
"""

UE coordinate system: (left handed)
    x : front
    y : right
    z : up
Roation order: xyz
    roll : x
    pitch: y
    yaw  : z

"""
# ================ Camera Params =================
def save_camera(frame_list, hfov_deg, channels, width=1280, height=720):

    def cache_channel_keys(channel):
        """
        一次性获取channel的所有keys并缓存为简单的(frame, value)列表
        避免在循环中重复调用get_keys()造成UObject泄漏
        """
        if channel is None:
            return []
        
        keys = channel.get_keys()
        if not keys:
            return []
        
        # 转换为简单的Python数据结构，释放UE对象引用
        cached_keys = [k.get_time(unreal.MovieSceneTimeUnit.DISPLAY_RATE).frame_number.value for k in keys]
        cached_values = [k.get_value() for k in keys]
        return cached_keys, cached_values
    
    def merge_channels(channel_x, channel_y, channel_z):
        cached_keys_x, cached_values_x = cache_channel_keys(channel_x)
        cached_keys_y, cached_values_y = cache_channel_keys(channel_y)
        cached_keys_z, cached_values_z = cache_channel_keys(channel_z)
        for x,y,z in zip(cached_keys_x,cached_keys_y,cached_keys_z):
            assert x==y and y==z
        return cached_keys_x, list(zip(cached_values_x,cached_values_y,cached_values_z))

    key_frames, key_frames_loc = merge_channels(channels[0], channels[1], channels[2]) # x,y,z
    key_frames, key_frames_rot = merge_channels(channels[3], channels[4], channels[5]) # roll, pitch, yaw
    
    def evaluate_channel(channel, target_frame_number):
        keys = channel.get_keys()
        if not keys:
            return channel.get_default() if channel.has_default() else 0.0

        # Sort keys by time
        sorted_keys = sorted(keys, key=lambda k: k.get_time(unreal.MovieSceneTimeUnit.DISPLAY_RATE).frame_number.value)

        target = target_frame_number.value

        for idx in range(1, len(sorted_keys)):
            left = sorted_keys[idx - 1]
            right = sorted_keys[idx]
            left_time = left.get_time(unreal.MovieSceneTimeUnit.DISPLAY_RATE).frame_number.value
            right_time = right.get_time(unreal.MovieSceneTimeUnit.DISPLAY_RATE).frame_number.value

            if left_time <= target <= right_time:
                if left_time == right_time:
                    return left.get_value()
                fraction = (target - left_time) / (right_time - left_time)
                return left.get_value() + fraction * (right.get_value() - left.get_value())

        # Before first key
        if target < sorted_keys[0].get_time(unreal.MovieSceneTimeUnit.DISPLAY_RATE).frame_number.value:
            return sorted_keys[0].get_value()  # or handle pre-extrapolation

        # After last key
        return sorted_keys[-1].get_value()  # or handle post-extrapolation

    def evaluate_vector_cached(key_frames, key_frames_vector, target_frame_number, is_rotation=False):
        
        # Sort keys by time
        target = target_frame_number.value

        for idx in range(1, len(key_frames)):
            left_time = key_frames[idx - 1]
            right_time = key_frames[idx]
            left_v = key_frames_vector[idx-1]
            right_v = key_frames_vector[idx]
            if left_time <= target <= right_time:
                if left_time==right_time:
                    return left_v
                fraction = (target - left_time) / (right_time - left_time)
                if is_rotation:
                    left_rot = unreal.Rotator(*left_v)
                    right_rot = unreal.Rotator(*right_v)
                    res_quat = left_rot.quaternion().slerp_quat(right_rot.quaternion(), fraction)
                    res_rot = res_quat.rotator()
                    return [res_rot.roll, res_rot.pitch, res_rot.yaw]
                else:
                    return [left + fraction * (right - left) for left,right in zip(left_v, right_v)]
        # Before first key
        if target < key_frames[0]:
            return key_frames_vector[0]  # or handle pre-extrapolation

        # After last key
        return key_frames_vector[-1]  # or handle post-extrapolation

    # Calculate intrinsics
    print("[INFO][CameraSave] hfov_deg=",hfov_deg)
    cx = width / 2.0
    cy = height / 2.0
    intrinsics = {
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "hfov_deg": hfov_deg
    }

    # Calculate extrinsics (poses) by evaluating channels for each frame
    poses = []
    for i,frame in enumerate(frame_list):
        frame_number = unreal.FrameNumber(frame)
        # loc_x = evaluate_channel(channels[0], frame_number)
        # loc_y = evaluate_channel(channels[1], frame_number)
        # loc_z = evaluate_channel(channels[2], frame_number)
        loc_x, loc_y, loc_z = evaluate_vector_cached(key_frames, key_frames_loc, frame_number)
        roll, pitch, yaw = evaluate_vector_cached(key_frames, key_frames_rot, frame_number, is_rotation=True)
        # roll = evaluate_channel(channels[3], frame_number)
        # pitch = evaluate_channel(channels[4], frame_number)
        # yaw = evaluate_channel(channels[5], frame_number)
        poses.append({
            "frame": i,
            "idx": frame,
            "location": [loc_x, loc_y, loc_z],
            "rotation": {
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll
            }
        })

    # Save to JSON
    data = {
        "intrinsics": intrinsics,
        "poses": poses
    }
    return data

def get_camera_pose_pan(num_key_frames, init_location, init_rot_vec, target_translation):
    """
    generate camera sequence, start from frame '1'
    return a sequnce with length (num_key_frames+1)
    """
    key_frame_locations = []
    key_frame_rotations = []
    for i in range(0, num_key_frames):
        
        trans = (target_translation * (i) / (num_key_frames-1))
        # new location to keep look at point unchanged
        # trans = target_translation
        location = init_location + trans
        rotation = init_rot_vec
        key_frame_locations.append(location)
        key_frame_rotations.append(rotation)
        # avoid yaw reach +- 90
    return key_frame_locations, key_frame_rotations

def get_camera_pose(num_key_frames, init_location, init_rot_vec, radius, target_rotation, target_radius):
    """
    generate camera sequence, start from frame '1'
    return a sequnce with length (num_key_frames+1)
    """
    key_frame_locations = []
    key_frame_rotations = []
    # radius = math.sqrt((init_location.x - center.x)**2 + (init_location.y - center.y)**2)
    for i in range(0, num_key_frames):
        direction = [radius, 0, 0]
        rel_rotation = target_rotation * (i) / (num_key_frames-1)
        # rel_rotation = target_rotation
        rel_R = lib_math.euler_to_rotation_matrix(init_rot_vec+rel_rotation)
        init_R = lib_math.euler_to_rotation_matrix(init_rot_vec)
        # direction is where lookat point rotated to
        lookat = lib_math.matrix_vector_multiply(init_R, direction )
        scaled_lookat = [(target_radius * (i) / (num_key_frames-1) + (num_key_frames-1-i) / (num_key_frames-1)) * radius, 0, 0]
        # scaled_lookat = [target_radius * radius, 0, 0]
        rotated_lookat = lib_math.matrix_vector_multiply(rel_R, scaled_lookat)
        # new location to keep look at point unchanged
        location = init_location + unreal.Vector(*lookat) - unreal.Vector(*rotated_lookat)
        rotation = init_rot_vec + rel_rotation
        key_frame_locations.append(location)
        key_frame_rotations.append(rotation)
        # avoid yaw reach +- 90
    return key_frame_locations, key_frame_rotations

def edit_channel(channels, frame, key_frame_location, key_frame_rotation, scale):
    frame = unreal.FrameNumber(frame)
    # Add keys to channels with linear interpolation
    interpolation = unreal.MovieSceneKeyInterpolation.LINEAR
    channels[0].add_key(frame, key_frame_location.x, interpolation=interpolation)  # Location X
    channels[1].add_key(frame, key_frame_location.y, interpolation=interpolation)  # Location Y
    channels[2].add_key(frame, key_frame_location.z, interpolation=interpolation)  # Location Z
    channels[3].add_key(frame, key_frame_rotation.x, interpolation=interpolation)  # Rotation Roll
    channels[4].add_key(frame, key_frame_rotation.y, interpolation=interpolation)  # Rotation Pitch
    channels[5].add_key(frame, key_frame_rotation.z, interpolation=interpolation)  # Rotation Yaw
    channels[6].add_key(frame, scale, interpolation=interpolation)  # Scale X
    channels[7].add_key(frame, scale, interpolation=interpolation)  # Scale Y
    channels[8].add_key(frame, scale, interpolation=interpolation)  # Scale Z

