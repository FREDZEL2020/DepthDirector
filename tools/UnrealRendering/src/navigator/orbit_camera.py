import unreal
import math
import time
import json
import os
import random
import datetime
from navigator.navigator import Navigator, SceneTrajectoryInfo, TransformationSequence
from libs.cameras import get_camera_pose
import libs.collison as lib_collison
import libs.math as lib_math
from params import MAX_ATTEMPTS
from navigator.navigator_navmesh import NavigatorNavMesh

class OrbitCamera(NavigatorNavMesh):
    def __init__(self, navigator_config, collison_config):
        super().__init__(navigator_config, collison_config)
        self.camera_type = ['center', 'rand', 'rand_input', 'double',
                           'orbit', 'helix_in', 's_curve_dolly', 'zoom_out', 'orbit_ascend', 'parabola_dive',
                           'pan_left_right', 'rotate_left_right']
    
    def get_cameras(self, cam_name, scale, initial_location, initial_rotation_vector, radius, start_frame):
        target_rotation = unreal.Vector(0, -30, -60)
        target_radius = 1.0
        length = self.frame_num + self.warmup_frame_num
        for i in range(MAX_ATTEMPTS):
            if cam_name.startswith("rand"):
                target_rotation = unreal.Vector(0, random.randint(self.navigator_config['y_min'],self.navigator_config['y_max']),(1 if (random.random() > 0.5) else -1 ) * random.randint(self.navigator_config.get('z_rot_min', 30), self.navigator_config.get('z_rot', 60)))
                target_radius = random.uniform(self.navigator_config['zoom_in'], self.navigator_config['zoom_out'])
                if cam_name == "rand_input":
                    target_rotation = unreal.Vector(0, random.randint(-5,5), random.randint(-10,10))
                    target_radius = random.uniform(self.navigator_config['zoom_in'], self.navigator_config['zoom_out'])
            elif cam_name == "center":
                target_rotation = unreal.Vector(0, 0, 0)
            elif cam_name.startswith("fix"):
                target_rotation = unreal.Vector(0, 0, int(cam_name.removeprefix("fix")))
            if cam_name.startswith("double"):
                num_keys = 2
                step = random.randint(30,50)
                target_rotation = unreal.Vector(0, random.randint(self.navigator_config['y_min'],self.navigator_config['y_max']), random.randint(-20,20))
                target_radius = random.uniform(self.navigator_config['zoom_in'], self.navigator_config['zoom_out'])
                target_rotation2 = unreal.Vector(0, random.randint(self.navigator_config['y_min'],self.navigator_config['y_max']), random.randint(-20,20))
                target_radius2 = random.uniform(self.navigator_config['zoom_in'], self.navigator_config['zoom_out'])
                key_frame_locations1, key_frame_rotations1 = get_camera_pose(num_keys, initial_location, initial_rotation_vector, radius, target_rotation, target_radius)
            
                key_frame_locations2, key_frame_rotations2 = get_camera_pose(num_keys, initial_location, initial_rotation_vector, radius, target_rotation2, target_radius2)
                print("[DEBUG][OrbitCamera]", key_frame_rotations1, key_frame_rotations2)
                key_frames = [0, step, self.frame_num-1]
                key_frame_locations = key_frame_locations1 + key_frame_locations2[1:]
                key_frame_rotations = key_frame_rotations1 + key_frame_rotations2[1:]
            # --- 新增轨迹逻辑分支 ---
            elif cam_name in ("orbit", "helix_in", "s_curve_dolly", "zoom_out", "orbit_ascend", "parabola_dive"):
                num_keys = 24
                step = (self.frame_num - 1) / (num_keys - 1)
                key_frames = [int(step * j) for j in range(num_keys)]

                initial_rot = unreal.Rotator(initial_rotation_vector.x, initial_rotation_vector.y, initial_rotation_vector.z)
                right = initial_rot.get_right_vector()
                up = initial_rot.get_up_vector()
                fwd = initial_rot.get_forward_vector()

                direction_mult = random.choice([1.0, -1.0])
                outdoor_scale = self.navigator_config.get('outdoor_scale', 1.0)
                R = radius

                key_frame_locations = []
                key_frame_rotations = []
                character_location = initial_location + fwd * radius

                if cam_name == "orbit":
                    rand_orbit_rad = R * random.uniform(0.1, 0.4)
                    rand_angle_scale = random.uniform(1.2, 1.8)
                elif cam_name == "helix_in":
                    rand_push_dist = R * random.uniform(0.1, 0.4)
                    rand_spiral_rad = R * random.uniform(0.02, 0.12)
                    rand_freq = random.uniform(2.0, 2.8) * math.pi
                elif cam_name == "s_curve_dolly":
                    rand_side_dist = R * random.uniform(0.1, 0.4)
                    rand_depth_amp = R * random.uniform(0.02, 0.12)
                elif cam_name == "zoom_out":
                    rand_zoom_dist = R * random.uniform(0.1, 0.4)
                elif cam_name == "orbit_ascend":
                    rand_swing_rad = R * random.uniform(0.1, 0.4)
                    rand_depth_scale = R * random.uniform(0.02, 0.08)
                    rand_up_scale = R * random.uniform(0.1, 0.4)
                elif cam_name == "parabola_dive":
                    rand_slide_dist = R * random.uniform(0.1, 0.4)
                    rand_drop_height = R * random.uniform(0.1, 0.4)
                    rand_push_scale = R * random.uniform(0.02, 0.12)

                for j in range(num_keys):
                    t = j / (num_keys - 1)

                    if cam_name == "orbit":
                        angle = t * rand_angle_scale * math.pi
                        off_r = rand_orbit_rad * (math.cos(angle) - 1.0)
                        off_u = rand_orbit_rad * math.sin(angle)
                        current_loc = initial_location + (right * off_r * direction_mult) + (up * off_u)

                    elif cam_name == "helix_in":
                        vec_push = fwd * rand_push_dist * t
                        vec_spiral = (right * (math.cos(t * rand_freq) - 1.0) * rand_spiral_rad * direction_mult) + \
                                     (up * math.sin(t * rand_freq) * rand_spiral_rad)
                        current_loc = initial_location + vec_push + vec_spiral

                    elif cam_name == "s_curve_dolly":
                        vec_side = right * rand_side_dist * t * direction_mult
                        vec_depth = fwd * math.sin(t * 2 * math.pi) * rand_depth_amp
                        current_loc = initial_location + vec_side + vec_depth

                    elif cam_name == "zoom_out":
                        current_loc = initial_location - (fwd * rand_zoom_dist * t)

                    elif cam_name == "orbit_ascend":
                        vec_swing = right * math.sin(t * math.pi) * rand_swing_rad * direction_mult
                        vec_depth = fwd * math.sin(t * 2 * math.pi) * rand_depth_scale
                        vec_up = up * rand_up_scale * t
                        current_loc = initial_location + vec_swing + vec_depth + vec_up

                    elif cam_name == "parabola_dive":
                        vec_slide = right * rand_slide_dist * t * direction_mult
                        vec_height = up * (math.cos(t * 2 * math.pi) - 1.0) * 0.5 * rand_drop_height
                        vec_push = fwd * math.sin(t * math.pi) * rand_push_scale
                        current_loc = initial_location + vec_slide + vec_height + vec_push

                    look_at_rot = unreal.MathLibrary.find_look_at_rotation(current_loc, character_location)
                    key_frame_locations.append(current_loc)
                    key_frame_rotations.append(lib_math.rotator_to_vec(look_at_rot))

            elif cam_name == "pan_left_right":
                num_keys = 24
                step = (self.frame_num - 1) / (num_keys - 1)
                key_frames = [int(step * j) for j in range(num_keys)]

                initial_rot = unreal.Rotator(initial_rotation_vector.x, initial_rotation_vector.y, initial_rotation_vector.z)
                right = initial_rot.get_right_vector()

                direction_mult = random.choice([1.0, -1.0])
                R = radius
                pan_dist = R * random.uniform(0.2, 0.5)

                key_frame_locations = []
                key_frame_rotations = []

                for j in range(num_keys):
                    t = j / (num_keys - 1)
                    current_loc = initial_location + right * pan_dist * t * direction_mult
                    key_frame_locations.append(current_loc)
                    key_frame_rotations.append(initial_rotation_vector)

            elif cam_name == "rotate_left_right":
                num_keys = 24
                step = (self.frame_num - 1) / (num_keys - 1)
                key_frames = [int(step * j) for j in range(num_keys)]

                direction_mult = random.choice([1.0, -1.0])
                yaw_range = random.uniform(20.0, 40.0)

                key_frame_locations = []
                key_frame_rotations = []

                for j in range(num_keys):
                    t = j / (num_keys - 1)
                    key_frame_locations.append(initial_location)
                    yaw_offset = yaw_range * t * direction_mult
                    rot = unreal.Vector(
                        initial_rotation_vector.x,
                        initial_rotation_vector.y,
                        initial_rotation_vector.z + yaw_offset
                    )
                    key_frame_rotations.append(rot)

            else:
                if random.randint(0,10) <= 5:
                    num_keys = 9
                    step = (self.frame_num-1) // (num_keys-1)
                else:
                    num_keys = 2
                    step = self.frame_num-1

                key_frame_locations, key_frame_rotations = get_camera_pose(num_keys, initial_location, initial_rotation_vector, radius, target_rotation, target_radius)
                key_frames = [step*i for i in range(num_keys)]

            valid = True
            last_loc = None
            for loc,rot in zip(key_frame_locations,key_frame_rotations):
                success, reason = self.validate_camera_pose(loc, rot, None)
                if not success:
                    valid = False
                    print("[FAIL][OrbitCamera] Checking Orbit Camera fail at attempt", i, cam_name, loc, rot, reason)
                    break
                if last_loc is not None:
                    target_dis = last_loc.distance(loc)
                    hit_distance = lib_collison.get_hit_location_sphere(last_loc, (loc-last_loc).normal(), distance=target_dis, SAFE_VOLUME_RADIUS=10)
                    if hit_distance < target_dis:
                        print(f"[FAIL][OrbitCamera] No straight path between: ", loc, f"Hit distance", hit_distance)
                        valid = False
                        continue
                last_loc = loc
            if valid:
                break
        if not valid:
            raise Exception("No valid camera for Orbit Camera", cam_name)
        key_frames[-1] = self.frame_num - 1
        key_frames = [0, self.warmup_frame_num] + [self.warmup_frame_num + i for i in key_frames]
        key_frame_locations = [initial_location, initial_location] + key_frame_locations
        key_frame_rotations = [initial_rotation_vector, initial_rotation_vector] + key_frame_rotations
        frame_info = SceneTrajectoryInfo( start_frame, length)
        frame_info.camera = TransformationSequence(key_frames, key_frame_locations, key_frame_rotations)
        return frame_info
