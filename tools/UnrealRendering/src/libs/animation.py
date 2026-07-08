"""
Animation utilities — skeletal-mesh / AnimSequence analysis helpers.

Primary use case: estimate the "native" walking speed of an AnimSequence at
play_rate=1, so callers (e.g. load_character_to_sequence) can compute the
play_rate required for a desired travel distance / speed.
"""

import math
import unreal
import libs.skeleton as lib_skeleton 

def estimate_walking_speed(skeletal_mesh, anim_sequence, foot="both", z_vel_threshold_cms=30.0):
    """Estimate walking speed by measuring instantaneous XY velocity at stance mid-point.

    Finds frames where foot is most stable (Z velocity near zero, XY velocity minimal),
    then averages the XY velocity at those frames to get walking speed.

    Method:
      1. Sample foot bones every frame in world space
      2. Compute Z velocity and XY velocity per frame
      3. Find stance frames: Z_vel <= threshold
      4. Within stance, find frame with minimum XY velocity (most stable contact)
      5. Walking speed = XY velocity at that frame

    Args:
        skeletal_mesh: provides bone proportions for pose evaluation
        anim_sequence: walk animation to analyze
        foot: "left", "right", or "both" (default: "both")
        z_vel_threshold_cms: Z velocity threshold for stance detection (default 30.0 cm/s)

    Returns:
        dict with keys:
            num_frames, duration, fps,
            foot_mode, left_bone, right_bone,
            left_stable_frame, right_stable_frame,
            left_xy_vel_cms, right_xy_vel_cms,
            walking_speed_cms, walking_speed_ms.
        Returns None on failure.
    """
    if skeletal_mesh is None or anim_sequence is None:
        unreal.log_error("[animation] estimate_walking_speed: missing mesh or anim.")
        return None

    num_frames = unreal.AnimationLibrary.get_num_frames(anim_sequence)
    duration = unreal.AnimationLibrary.get_sequence_length(anim_sequence)
    if num_frames <= 1 or duration <= 0.0:
        unreal.log_error(f"[animation] Invalid animation length: frames={num_frames}, duration={duration}")
        return None
    fps = (num_frames - 1) / duration
    dt = 1.0 / fps

    eval_opts = lib_skeleton._build_eval_options(skeletal_mesh)
    pose0 = unreal.AnimPoseExtensions.get_anim_pose_at_frame(anim_sequence, 0, eval_opts)
    if not unreal.AnimPoseExtensions.is_valid(pose0):
        unreal.log_error("[animation] Pose invalid at frame 0.")
        return None
    bone_names = [str(n) for n in unreal.AnimPoseExtensions.get_bone_names(pose0)]

    left_bone = lib_skeleton._pick_bone(bone_names, lib_skeleton.LEFT_FOOT_CANDIDATES, "l")
    right_bone = lib_skeleton._pick_bone(bone_names, lib_skeleton.RIGHT_FOOT_CANDIDATES, "r")

    mode = foot.lower()
    need_left = mode in ("left", "both")
    need_right = mode in ("right", "both")
    if need_left and not left_bone:
        unreal.log_error(f"[animation] Left foot bone not found.")
        return None
    if need_right and not right_bone:
        unreal.log_error(f"[animation] Right foot bone not found.")
        return None

    left_positions = [] if need_left else None
    right_positions = [] if need_right else None
    for f in range(num_frames):
        pose = unreal.AnimPoseExtensions.get_anim_pose_at_frame(anim_sequence, f, eval_opts)
        if need_left:
            lt = unreal.AnimPoseExtensions.get_bone_pose(pose, left_bone, unreal.AnimPoseSpaces.WORLD)
            left_positions.append(lt.translation)
        if need_right:
            rt = unreal.AnimPoseExtensions.get_bone_pose(pose, right_bone, unreal.AnimPoseSpaces.WORLD)
            right_positions.append(rt.translation)

    def _find_stable_stance_velocity(positions):
        """Compute median frame-to-frame XY speed within stance (Z vel near zero)."""
        if not positions or len(positions) < 3:
            return None, None
        n = len(positions)

        z_vel = [0.0] * n
        xy_vel = [0.0] * n
        for i in range(n):
            if i == 0:
                z_vel[i] = abs(positions[1].z - positions[0].z) / dt
                xy_vel[i] = math.hypot(positions[1].x - positions[0].x,
                                        positions[1].y - positions[0].y) / dt
            elif i == n - 1:
                z_vel[i] = abs(positions[n-1].z - positions[n-2].z) / dt
                xy_vel[i] = math.hypot(positions[n-1].x - positions[n-2].x,
                                        positions[n-1].y - positions[n-2].y) / dt
            else:
                z_vel[i] = abs(positions[i+1].z - positions[i-1].z) / (2 * dt)
                xy_vel[i] = math.hypot(positions[i+1].x - positions[i-1].x,
                                        positions[i+1].y - positions[i-1].y) / (2 * dt)

        stance_frames = [i for i in range(n) if z_vel[i] <= z_vel_threshold_cms]
        if not stance_frames:
            return None, None

        stance_xy_vels = sorted([xy_vel[i] for i in stance_frames])
        mid = len(stance_xy_vels) // 2
        median_vel = stance_xy_vels[mid] if len(stance_xy_vels) % 2 == 1 \
            else (stance_xy_vels[mid - 1] + stance_xy_vels[mid]) / 2.0

        return len(stance_frames), median_vel

    left_stance_count, left_median_vel = _find_stable_stance_velocity(left_positions) if need_left else (None, None)
    right_stance_count, right_median_vel = _find_stable_stance_velocity(right_positions) if need_right else (None, None)

    if mode == "left":
        walking_speed_cms = left_median_vel if left_median_vel is not None else 0.0
    elif mode == "right":
        walking_speed_cms = right_median_vel if right_median_vel is not None else 0.0
    else:
        valid_vels = [v for v in [left_median_vel, right_median_vel] if v is not None]
        walking_speed_cms = sum(valid_vels) / len(valid_vels) if valid_vels else 0.0

    result = {
        "num_frames": num_frames,
        "duration": duration,
        "fps": fps,
        "foot_mode": mode,
        "left_bone": left_bone,
        "right_bone": right_bone,
        "left_stance_frames": left_stance_count,
        "right_stance_frames": right_stance_count,
        "left_xy_vel_cms": left_median_vel,
        "right_xy_vel_cms": right_median_vel,
        "walking_distance_cm": walking_speed_cms * duration,
        "walking_speed_cms": walking_speed_cms,
        "walking_speed_ms": walking_speed_cms / 100.0,
    }
    left_vel_str = f"{left_median_vel:.2f}" if left_median_vel is not None else "N/A"
    right_vel_str = f"{right_median_vel:.2f}" if right_median_vel is not None else "N/A"
    unreal.log(
        f"[animation] walk speed (stance median velocity): "
        f"L={left_stance_count} frames vel={left_vel_str}cm/s, "
        f"R={right_stance_count} frames vel={right_vel_str}cm/s, "
        f"avg_speed={walking_speed_cms:.2f}cm/s"
    )
    return result


def estimate_walking_speed_bk(skeletal_mesh, anim_sequence, foot="both"):
    """[BACKUP] Estimate walking speed via cumulative per-frame foot displacement.

    The method:
      1. Sample one (or both) foot bones every frame in component (WORLD) space.
      2. Sum horizontal |Δxy| between consecutive frames → total foot path.
      3. walking_distance = total path (for an in-place walk, the foot sweeps
         forward during swing and backward during stance; the two combined
         per gait cycle equal the body's per-cycle forward travel).
      4. walking_speed = walking_distance / duration.

    Args:
        skeletal_mesh (unreal.SkeletalMesh): provides bone proportions for
            pose evaluation.
        anim_sequence (unreal.AnimSequence): walk animation to analyze.
        foot (str): "left", "right", or "both" (average of the two). Default "both".

    Returns:
        dict with keys:
            num_frames, duration, fps,
            foot_mode, left_bone, right_bone,
            left_path_cm, right_path_cm,
            walking_distance_cm, walking_speed_cms, walking_speed_ms.
        Returns None on failure.
    """
    if skeletal_mesh is None or anim_sequence is None:
        unreal.log_error("[animation] estimate_walking_speed_v2: missing mesh or anim.")
        return None

    num_frames = unreal.AnimationLibrary.get_num_frames(anim_sequence)
    duration = unreal.AnimationLibrary.get_sequence_length(anim_sequence)
    if num_frames <= 1 or duration <= 0.0:
        unreal.log_error(f"[animation] Invalid animation length: frames={num_frames}, duration={duration}")
        return None
    fps = (num_frames - 1) / duration

    eval_opts = lib_skeleton._build_eval_options(skeletal_mesh)
    pose0 = unreal.AnimPoseExtensions.get_anim_pose_at_frame(anim_sequence, 0, eval_opts)
    if not unreal.AnimPoseExtensions.is_valid(pose0):
        unreal.log_error("[animation] Pose invalid at frame 0.")
        return None
    bone_names = [str(n) for n in unreal.AnimPoseExtensions.get_bone_names(pose0)]

    left_bone = lib_skeleton._pick_bone(bone_names, lib_skeleton.LEFT_FOOT_CANDIDATES, "l")
    right_bone = lib_skeleton._pick_bone(bone_names, lib_skeleton.RIGHT_FOOT_CANDIDATES, "r")

    mode = foot.lower()
    need_left = mode in ("left", "both")
    need_right = mode in ("right", "both")
    if need_left and not left_bone:
        unreal.log_error(f"[animation] Left foot bone not found. Bones: {bone_names}")
        return None
    if need_right and not right_bone:
        unreal.log_error(f"[animation] Right foot bone not found. Bones: {bone_names}")
        return None

    left_positions = [] if need_left else None
    right_positions = [] if need_right else None
    for f in range(num_frames):
        pose = unreal.AnimPoseExtensions.get_anim_pose_at_frame(anim_sequence, f, eval_opts)
        if need_left:
            lt = unreal.AnimPoseExtensions.get_bone_pose(pose, left_bone, unreal.AnimPoseSpaces.WORLD)
            left_positions.append(lt.translation)
        if need_right:
            rt = unreal.AnimPoseExtensions.get_bone_pose(pose, right_bone, unreal.AnimPoseSpaces.WORLD)
            right_positions.append(rt.translation)

    def _path_len_xy(positions):
        total = 0.0
        for i in range(1, len(positions)):
            dx = positions[i].x - positions[i - 1].x
            dy = positions[i].y - positions[i - 1].y
            total += math.hypot(dx, dy)
        return total

    left_path_cm = _path_len_xy(left_positions) if need_left else None
    right_path_cm = _path_len_xy(right_positions) if need_right else None

    if mode == "left":
        walking_distance_cm = left_path_cm
    elif mode == "right":
        walking_distance_cm = right_path_cm
    else:
        walking_distance_cm = 0.5 * (left_path_cm + right_path_cm)

    walking_speed_cms = walking_distance_cm / duration

    result = {
        "num_frames": num_frames,
        "duration": duration,
        "fps": fps,
        "foot_mode": mode,
        "left_bone": left_bone,
        "right_bone": right_bone,
        "left_path_cm": left_path_cm,
        "right_path_cm": right_path_cm,
        "walking_distance_cm": walking_distance_cm,
        "walking_speed_cms": walking_speed_cms,
        "walking_speed_ms": walking_speed_cms / 100.0,
    }
    unreal.log(
        f"[animation] walk speed v2 ({mode}): "
        f"L_path={left_path_cm if left_path_cm is None else f'{left_path_cm:.2f}'}cm, "
        f"R_path={right_path_cm if right_path_cm is None else f'{right_path_cm:.2f}'}cm, "
        f"distance/anim={walking_distance_cm:.2f}cm, speed={walking_speed_cms:.2f}cm/s"
    )
    return result


def _bbox_extent(positions):
    """Given a list of unreal.Vector positions, return (extent_vec, diagonal_cm)
    where extent_vec = (max-min) per axis and diagonal = ||extent_vec||.
    Returns (None, None) if positions is empty.
    """
    if not positions:
        return None, None
    xs = [p.x for p in positions]
    ys = [p.y for p in positions]
    zs = [p.z for p in positions]
    ex = max(xs) - min(xs)
    ey = max(ys) - min(ys)
    ez = max(zs) - min(zs)
    return (ex, ey, ez), math.sqrt(ex * ex + ey * ey + ez * ez)


def get_anim_properties(anim_sequence, skeletal_mesh=None):
    """Query fundamental properties of an AnimSequence.

    Detects looping by comparing the first and last frame positions of the
    foot bones AND hand bones — if both feet and hands return to near-identical
    positions, the animation is a candidate for looping. Also samples every frame
    to compute the world-space movement range (axis-aligned bounding box) of the
    left hand, right hand, and character root bone.

    Args:
        anim_sequence (unreal.AnimSequence): animation asset to inspect.
        skeletal_mesh (unreal.SkeletalMesh, optional): provides bone proportions
            for pose evaluation. Required for loop detection.

    Returns:
        dict with keys:
            num_frames (int): total number of frames in the sequence.
            is_looping (bool): True if both foot and hand positions loop
                (foot_loop_match=True AND hand_loop_match=True or None).
            foot_loop_match (bool): True if first/last foot positions are
                within LOOP_THRESHOLD_CM.
            foot_first_pos / foot_last_pos (Vector): world-space foot positions
                at frame 0 and the last frame.
            foot_loop_distance_cm (float): distance between them, in cm.
            hand_loop_match (bool): True if both hands loop (or single hand if
                only one detected).
            left_hand_loop_match / right_hand_loop_match (bool): per-hand loop status.
            left_hand_first_pos / left_hand_last_pos (Vector): left hand positions.
            left_hand_loop_distance_cm (float): left hand first/last distance.
            right_hand_first_pos / right_hand_last_pos (Vector): right hand positions.
            right_hand_loop_distance_cm (float): right hand first/last distance.
            left_bone / right_bone (str): detected foot bone names.
            left_hand_bone / right_hand_bone / root_bone (str): detected bones
                used for range computation.
            left_hand_range_cm / right_hand_range_cm / root_range_cm (float):
                diagonal of the world-space bounding box swept by each bone
                across the whole animation (max movement range in cm).
            left_hand_extent / right_hand_extent / root_extent (tuple[float,float,float]):
                per-axis (x, y, z) extent of each bbox in cm.
        Returns None if anim_sequence is None or invalid.
    """
    if anim_sequence is None:
        unreal.log_error("[animation] get_anim_properties: anim_sequence is None.")
        return None

    foot_loop_match = None
    foot_first_pos = None
    foot_last_pos = None
    foot_loop_distance_cm = None
    left_bone = None
    right_bone = None
    left_hand_bone = None
    right_hand_bone = None
    root_bone = None
    left_hand_range_cm = None
    right_hand_range_cm = None
    root_range_cm = None
    left_hand_extent = None
    right_hand_extent = None
    root_extent = None

    num_frames = unreal.AnimationLibrary.get_num_frames(anim_sequence)
    eval_opts = lib_skeleton._build_eval_options(skeletal_mesh)
    pose0 = unreal.AnimPoseExtensions.get_anim_pose_at_frame(anim_sequence, 0, eval_opts)
    pose_last = unreal.AnimPoseExtensions.get_anim_pose_at_frame(anim_sequence, num_frames - 1, eval_opts)
    if not (unreal.AnimPoseExtensions.is_valid(pose0) and unreal.AnimPoseExtensions.is_valid(pose_last)):
        unreal.log_warning("[animation] Could not evaluate first/last poses, skipping loop detection.")
        return {
            "num_frames": num_frames,
            "is_looping": False,
            "foot_loop_match": False,
            "foot_first_pos": None,
            "foot_last_pos": None,
            "foot_loop_distance_cm": None,
            "hand_loop_match": None,
            "left_hand_loop_match": None,
            "left_hand_first_pos": None,
            "left_hand_last_pos": None,
            "left_hand_loop_distance_cm": None,
            "right_hand_loop_match": None,
            "right_hand_first_pos": None,
            "right_hand_last_pos": None,
            "right_hand_loop_distance_cm": None,
            "left_bone": None,
            "right_bone": None,
            "left_hand_bone": None,
            "right_hand_bone": None,
            "root_bone": None,
            "left_hand_range_cm": None,
            "right_hand_range_cm": None,
            "root_range_cm": None,
            "left_hand_extent": None,
            "right_hand_extent": None,
            "root_extent": None,
        }

    bone_names = [str(n) for n in unreal.AnimPoseExtensions.get_bone_names(pose0)]
    left_bone = lib_skeleton._pick_bone(bone_names, lib_skeleton.LEFT_FOOT_CANDIDATES, "l")
    right_bone = lib_skeleton._pick_bone(bone_names, lib_skeleton.RIGHT_FOOT_CANDIDATES, "r")
    left_hand_bone = lib_skeleton._pick_bone(bone_names, lib_skeleton.LEFT_HAND_CANDIDATES, "l",
                                             keyword="hand", bad_tokens=("handle",))
    right_hand_bone = lib_skeleton._pick_bone(bone_names, lib_skeleton.RIGHT_HAND_CANDIDATES, "r",
                                              keyword="hand", bad_tokens=("handle",))
    root_bone = lib_skeleton._pick_root_bone(bone_names)

    LOOP_THRESHOLD_CM = 25.0

    def _bone_loop_distance(pose_first, pose_last_frame, bone_name):
        if not bone_name:
            return None, None, None
        t0 = unreal.AnimPoseExtensions.get_bone_pose(pose_first, bone_name, unreal.AnimPoseSpaces.WORLD).translation
        t1 = unreal.AnimPoseExtensions.get_bone_pose(pose_last_frame, bone_name, unreal.AnimPoseSpaces.WORLD).translation
        dist = math.sqrt((t0.x - t1.x) ** 2 + (t0.y - t1.y) ** 2 + (t0.z - t1.z) ** 2)
        return t0, t1, dist

    foot_bone = left_bone or right_bone
    if foot_bone:
        foot_first_pos, foot_last_pos, foot_loop_distance_cm = _bone_loop_distance(pose0, pose_last, foot_bone)
        foot_loop_match = foot_loop_distance_cm < LOOP_THRESHOLD_CM
    else:
        unreal.log_warning(f"[animation] Foot bones not found. Bones: {bone_names}")

    left_hand_first_pos = None
    left_hand_last_pos = None
    left_hand_loop_distance_cm = None
    left_hand_loop_match = None
    right_hand_first_pos = None
    right_hand_last_pos = None
    right_hand_loop_distance_cm = None
    right_hand_loop_match = None

    if left_hand_bone:
        left_hand_first_pos, left_hand_last_pos, left_hand_loop_distance_cm = _bone_loop_distance(pose0, pose_last, left_hand_bone)
        left_hand_loop_match = left_hand_loop_distance_cm < LOOP_THRESHOLD_CM
    if right_hand_bone:
        right_hand_first_pos, right_hand_last_pos, right_hand_loop_distance_cm = _bone_loop_distance(pose0, pose_last, right_hand_bone)
        right_hand_loop_match = right_hand_loop_distance_cm < LOOP_THRESHOLD_CM

    hand_loop_match = None
    if left_hand_loop_match is not None and right_hand_loop_match is not None:
        hand_loop_match = left_hand_loop_match and right_hand_loop_match
    elif left_hand_loop_match is not None:
        hand_loop_match = left_hand_loop_match
    elif right_hand_loop_match is not None:
        hand_loop_match = right_hand_loop_match

    is_looping = bool(foot_loop_match) and (hand_loop_match is None or hand_loop_match)

    # Sample every frame once; compute bbox for each tracked bone.
    tracked = [(name, []) for name in (left_hand_bone, right_hand_bone, root_bone) if name]
    if tracked:
        for f in range(num_frames):
            pose = unreal.AnimPoseExtensions.get_anim_pose_at_frame(anim_sequence, f, eval_opts)
            if not unreal.AnimPoseExtensions.is_valid(pose):
                continue
            for bone, positions in tracked:
                tf = unreal.AnimPoseExtensions.get_bone_pose(pose, bone, unreal.AnimPoseSpaces.WORLD)
                positions.append(tf.translation)
        by_bone = {bone: positions for bone, positions in tracked}
        if left_hand_bone:
            left_hand_extent, left_hand_range_cm = _bbox_extent(by_bone[left_hand_bone])
        if right_hand_bone:
            right_hand_extent, right_hand_range_cm = _bbox_extent(by_bone[right_hand_bone])
        if root_bone:
            root_extent, root_range_cm = _bbox_extent(by_bone[root_bone])

    def _fmt(v):
        return f"{v:.2f}" if v is not None else "n/a"

    unreal.log(
        f"[animation] is_looping={is_looping} | "
        f"foot_loop={foot_loop_match} (dist={_fmt(foot_loop_distance_cm)}cm via '{foot_bone}') | "
        f"hand_loop={hand_loop_match} (L={_fmt(left_hand_loop_distance_cm)}cm, R={_fmt(right_hand_loop_distance_cm)}cm) | "
        f"L_hand_range={_fmt(left_hand_range_cm)}cm ('{left_hand_bone}'), "
        f"R_hand_range={_fmt(right_hand_range_cm)}cm ('{right_hand_bone}'), "
        f"root_range={_fmt(root_range_cm)}cm ('{root_bone}')"
    )
    return {
        "num_frames": num_frames,
        "is_looping": is_looping,
        "foot_loop_match": foot_loop_match,
        "foot_first_pos": foot_first_pos,
        "foot_last_pos": foot_last_pos,
        "foot_loop_distance_cm": foot_loop_distance_cm,
        "hand_loop_match": hand_loop_match,
        "left_hand_loop_match": left_hand_loop_match,
        "left_hand_first_pos": left_hand_first_pos,
        "left_hand_last_pos": left_hand_last_pos,
        "left_hand_loop_distance_cm": left_hand_loop_distance_cm,
        "right_hand_loop_match": right_hand_loop_match,
        "right_hand_first_pos": right_hand_first_pos,
        "right_hand_last_pos": right_hand_last_pos,
        "right_hand_loop_distance_cm": right_hand_loop_distance_cm,
        "left_bone": left_bone,
        "right_bone": right_bone,
        "left_hand_bone": left_hand_bone,
        "right_hand_bone": right_hand_bone,
        "root_bone": root_bone,
        "left_hand_range_cm": left_hand_range_cm,
        "right_hand_range_cm": right_hand_range_cm,
        "root_range_cm": root_range_cm,
        "left_hand_extent": left_hand_extent,
        "right_hand_extent": right_hand_extent,
        "root_extent": root_extent,
    }


def estimate_blended_walking_speed(skeletal_mesh, walk_anim, idle_anim, walk_weight, foot="both", z_vel_threshold_cms=30.0):
    """估算混合动画（walk * walk_weight + idle * (1 - walk_weight)）的行走速度。

    找到最稳定的着地帧（Z 速度≈0，XY 速度最小），测量该帧的 XY 瞬时速度。

    Args:
        skeletal_mesh: 骨骼网格
        walk_anim: walk 动画
        idle_anim: idle 动画
        walk_weight: walk 动画权重（0.0-1.0）
        foot: "left", "right", "both"
        z_vel_threshold_cms: Z 速度阈值 (default 30.0 cm/s)

    Returns:
        dict 包含 walking_speed_cms 等字段，失败返回 None
    """
    if skeletal_mesh is None or walk_anim is None or idle_anim is None:
        unreal.log_error("[animation] estimate_blended_walking_speed: missing mesh or anim.")
        return None

    num_frames = unreal.AnimationLibrary.get_num_frames(walk_anim)
    duration = unreal.AnimationLibrary.get_sequence_length(walk_anim)
    if num_frames <= 1 or duration <= 0.0:
        unreal.log_error(f"[animation] Invalid walk animation length: frames={num_frames}, duration={duration}")
        return None
    fps = (num_frames - 1) / duration
    dt = 1.0 / fps

    idle_duration = unreal.AnimationLibrary.get_sequence_length(idle_anim)
    idle_num_frames = unreal.AnimationLibrary.get_num_frames(idle_anim)
    if idle_num_frames <= 1 or idle_duration <= 0.0:
        unreal.log_error(f"[animation] Invalid idle animation length")
        return None

    eval_opts = lib_skeleton._build_eval_options(skeletal_mesh)
    pose0 = unreal.AnimPoseExtensions.get_anim_pose_at_frame(walk_anim, 0, eval_opts)
    if not unreal.AnimPoseExtensions.is_valid(pose0):
        unreal.log_error("[animation] Pose invalid at frame 0.")
        return None
    bone_names = [str(n) for n in unreal.AnimPoseExtensions.get_bone_names(pose0)]

    left_bone = lib_skeleton._pick_bone(bone_names, lib_skeleton.LEFT_FOOT_CANDIDATES, "l")
    right_bone = lib_skeleton._pick_bone(bone_names, lib_skeleton.RIGHT_FOOT_CANDIDATES, "r")

    mode = foot.lower()
    need_left = mode in ("left", "both")
    need_right = mode in ("right", "both")
    if need_left and not left_bone:
        unreal.log_error(f"[animation] Left foot bone not found.")
        return None
    if need_right and not right_bone:
        unreal.log_error(f"[animation] Right foot bone not found.")
        return None

    idle_weight = 1.0 - walk_weight

    left_positions = [] if need_left else None
    right_positions = [] if need_right else None
    for f in range(num_frames):
        walk_pose = unreal.AnimPoseExtensions.get_anim_pose_at_frame(walk_anim, f, eval_opts)
        normalized_time = f / (num_frames - 1) if num_frames > 1 else 0.0
        idle_frame = int(normalized_time * (idle_num_frames - 1))
        idle_pose = unreal.AnimPoseExtensions.get_anim_pose_at_frame(idle_anim, idle_frame, eval_opts)

        if need_left:
            walk_lt = unreal.AnimPoseExtensions.get_bone_pose(walk_pose, left_bone, unreal.AnimPoseSpaces.WORLD).translation
            idle_lt = unreal.AnimPoseExtensions.get_bone_pose(idle_pose, left_bone, unreal.AnimPoseSpaces.WORLD).translation
            blended_lt = unreal.Vector(
                walk_lt.x * walk_weight + idle_lt.x * idle_weight,
                walk_lt.y * walk_weight + idle_lt.y * idle_weight,
                walk_lt.z * walk_weight + idle_lt.z * idle_weight
            )
            left_positions.append(blended_lt)
        if need_right:
            walk_rt = unreal.AnimPoseExtensions.get_bone_pose(walk_pose, right_bone, unreal.AnimPoseSpaces.WORLD).translation
            idle_rt = unreal.AnimPoseExtensions.get_bone_pose(idle_pose, right_bone, unreal.AnimPoseSpaces.WORLD).translation
            blended_rt = unreal.Vector(
                walk_rt.x * walk_weight + idle_rt.x * idle_weight,
                walk_rt.y * walk_weight + idle_rt.y * idle_weight,
                walk_rt.z * walk_weight + idle_rt.z * idle_weight
            )
            right_positions.append(blended_rt)

    def _find_stable_stance_velocity(positions):
        """Compute median frame-to-frame XY speed within stance (Z vel near zero)."""
        if not positions or len(positions) < 3:
            return None, None
        n = len(positions)

        z_vel = [0.0] * n
        xy_vel = [0.0] * n
        for i in range(n):
            if i == 0:
                z_vel[i] = abs(positions[1].z - positions[0].z) / dt
                xy_vel[i] = math.hypot(positions[1].x - positions[0].x,
                                        positions[1].y - positions[0].y) / dt
            elif i == n - 1:
                z_vel[i] = abs(positions[n-1].z - positions[n-2].z) / dt
                xy_vel[i] = math.hypot(positions[n-1].x - positions[n-2].x,
                                        positions[n-1].y - positions[n-2].y) / dt
            else:
                z_vel[i] = abs(positions[i+1].z - positions[i-1].z) / (2 * dt)
                xy_vel[i] = math.hypot(positions[i+1].x - positions[i-1].x,
                                        positions[i+1].y - positions[i-1].y) / (2 * dt)

        stance_frames = [i for i in range(n) if z_vel[i] <= z_vel_threshold_cms]
        if not stance_frames:
            return None, None

        stance_xy_vels = sorted([xy_vel[i] for i in stance_frames])
        mid = len(stance_xy_vels) // 2
        median_vel = stance_xy_vels[mid] if len(stance_xy_vels) % 2 == 1 \
            else (stance_xy_vels[mid - 1] + stance_xy_vels[mid]) / 2.0

        return len(stance_frames), median_vel

    left_stance_count, left_median_vel = _find_stable_stance_velocity(left_positions) if need_left else (None, None)
    right_stance_count, right_median_vel = _find_stable_stance_velocity(right_positions) if need_right else (None, None)

    if mode == "left":
        walking_speed_cms = left_median_vel if left_median_vel is not None else 0.0
    elif mode == "right":
        walking_speed_cms = right_median_vel if right_median_vel is not None else 0.0
    else:
        valid_vels = [v for v in [left_median_vel, right_median_vel] if v is not None]
        walking_speed_cms = sum(valid_vels) / len(valid_vels) if valid_vels else 0.0

    result = {
        "num_frames": num_frames,
        "duration": duration,
        "fps": fps,
        "foot_mode": mode,
        "left_bone": left_bone,
        "right_bone": right_bone,
        "left_stance_frames": left_stance_count,
        "right_stance_frames": right_stance_count,
        "left_xy_vel_cms": left_median_vel,
        "right_xy_vel_cms": right_median_vel,
        "walking_distance_cm": walking_speed_cms * duration,
        "walking_speed_cms": walking_speed_cms,
        "walking_speed_ms": walking_speed_cms / 100.0,
        "walk_weight": walk_weight,
    }
    left_vel_str = f"{left_median_vel:.2f}" if left_median_vel is not None else "N/A"
    right_vel_str = f"{right_median_vel:.2f}" if right_median_vel is not None else "N/A"
    unreal.log(
        f"[animation] blended walk speed (w={walk_weight:.3f}, stance median velocity): "
        f"L={left_stance_count} frames vel={left_vel_str}cm/s, "
        f"R={right_stance_count} frames vel={right_vel_str}cm/s, "
        f"avg_speed={walking_speed_cms:.2f}cm/s"
    )
    return result


def compute_play_rate_for_distance(skeletal_mesh, anim_sequence, target_distance_cm, target_duration_s):
    """Compute the play_rate needed so the character covers `target_distance_cm`
    in `target_duration_s` seconds using this walk animation.

    play_rate > 1 speeds the animation up, < 1 slows it down. We set the
    animation's native speed * play_rate = required speed:
        required_speed = target_distance_cm / target_duration_s
        play_rate      = required_speed / native_walking_speed

    Returns (play_rate, stats_dict) or (None, None) on failure.
    """
    stats = estimate_walking_speed(skeletal_mesh, anim_sequence)
    if stats is None or stats["walking_speed_cms"] <= 0.0 or target_duration_s <= 0.0:
        return None, stats
    required_speed = target_distance_cm / target_duration_s
    play_rate = required_speed / stats["walking_speed_cms"]
    return play_rate, stats


if __name__ == "__main__":
    anim = unreal.load_asset("/Game/SchoolGirls/Demo/Animations/ThirdPersonWalk")
    skeleton_mesh = unreal.load_asset("/Game/SchoolGirls/Mesh/SK_SchoolGirl_v7")
    # props = get_anim_properties(anim, skeleton_mesh)
    # print(props)
    speed = estimate_walking_speed(skeleton_mesh, anim)
    print(speed)
    