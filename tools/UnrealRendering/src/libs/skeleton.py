import unreal
import json
import math
import os
import sys
import random



LEFT_FOOT_CANDIDATES = [
    "foot_l", "Foot_L", "LeftFoot", "left_foot", "L_Foot", "ball_l",
    "Bip01-L-Foot", "Bip01_L_Foot", "Bip01 L Foot",
]
RIGHT_FOOT_CANDIDATES = [
    "foot_r", "Foot_R", "RightFoot", "right_foot", "R_Foot", "ball_r",
    "Bip01-R-Foot", "Bip01_R_Foot", "Bip01 R Foot",
]
LEFT_HAND_CANDIDATES = [
    "hand_l", "Hand_L", "LeftHand", "left_hand", "L_Hand",
    "Bip01-L-Hand", "Bip01_L_Hand", "Bip01 L Hand",
]
RIGHT_HAND_CANDIDATES = [
    "hand_r", "Hand_R", "RightHand", "right_hand", "R_Hand",
    "Bip01-R-Hand", "Bip01_R_Hand", "Bip01 R Hand",
]
ROOT_CANDIDATES = [
    "root", "Root",
    "pelvis", "Pelvis",
    "Hips", "hips",
    "Bip01", "Bip01-Pelvis", "Bip01_Pelvis",
]


def _pick_bone(bone_names, candidates, side, keyword="foot",
               bad_tokens=("toe", "nub", "footstep")):
    """Pick the first candidate that exists in bone_names (case-insensitive).

    Falls back to any bone that contains `keyword` together with a side marker
    (_l_, -l-, etc.), skipping tokens listed in `bad_tokens`.
    """
    lower_map = {str(b).lower(): str(b) for b in bone_names}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    side_markers = [f"_{side}_", f"-{side}-", f" {side} ",
                    f"_{side}", f"-{side}", f"{side}_", f"{side}-"]
    best = None
    for name_lower, name in lower_map.items():
        if keyword not in name_lower:
            continue
        if any(bad in name_lower for bad in bad_tokens):
            continue
        if any(m in name_lower for m in side_markers):
            if best is None or len(name_lower) < len(best.lower()):
                best = name
    return best


def _pick_root_bone(bone_names):
    """Pick the character's root/pelvis bone.

    Tries ROOT_CANDIDATES first (case-insensitive), then falls back to the
    first bone in the list (which is the skeleton root by convention).
    """
    lower_map = {str(b).lower(): str(b) for b in bone_names}
    for c in ROOT_CANDIDATES:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return str(bone_names[0]) if bone_names else None


def _build_eval_options(skeletal_mesh):
    opts = unreal.AnimPoseEvaluationOptions()
    opts.set_editor_property("evaluation_type", unreal.AnimDataEvalType.SOURCE)
    opts.set_editor_property("should_retarget", False)
    opts.set_editor_property("extract_root_motion", False)
    opts.set_editor_property("incorporate_root_motion_into_pose", False)
    opts.set_editor_property("optional_skeletal_mesh", skeletal_mesh)
    opts.set_editor_property("retrieve_additive_as_full_pose", True)
    opts.set_editor_property("evaluate_curves", False)
    return opts

def estimate_skeletalmesh_property(skeletal_mesh):
    """估算 SkeletalMesh 在其自身 mesh-local（component）空间下的几何属性。

    Returns dict（失败返回 None）:
        forward_offset_deg: mesh-local +X 轴到人物正面朝前方向的夹角（度，绕 +Z，CCW 为正）。
        foot_height_cm:     脚部关节在 mesh-local 空间中的 Z 坐标（cm）。
        foot_bone / left_bone / right_bone: 在 reference pose 中挑选到的脚骨名。
        foot_pos_mesh:      脚骨在 mesh-local 空间的 (x, y, z)。
        forward_vec_mesh:   mesh-local 空间下人物正面朝前的单位向量 (x, y, 0)，
                            便于上层（如 blueprint）叠加 SkeletalMeshComponent 的相对变换。

    方法：在 Skeleton 的 reference pose 中用 _pick_bone 定位左右脚（参考
    animation.py），以 (right - left) 在 XY 面顺时针转 90° 作为 mesh 本地前进方向
    （UE 左手系 +X 前 / +Y 右 / +Z 上，forward × right = up）；脚骨的 Z 即脚高。
    """
    if skeletal_mesh is None:
        unreal.log_error("[skeleton] estimate_skeletalmesh_property: skeletal_mesh is None")
        return None
    skeleton = getattr(skeletal_mesh, "skeleton", None)
    if skeleton is None:
        unreal.log_error("[skeleton] estimate_skeletalmesh_property: no skeleton on mesh")
        return None

    ref_pose = skeleton.get_reference_pose()
    bone_names = [str(b) for b in ref_pose.get_bone_names()]
    left_bone = _pick_bone(bone_names, LEFT_FOOT_CANDIDATES, "l")
    right_bone = _pick_bone(bone_names, RIGHT_FOOT_CANDIDATES, "r")
    foot_bone = left_bone or right_bone
    if foot_bone is None:
        unreal.log_error(f"[skeleton] estimate_skeletalmesh_property: foot bone not found. Bones: {bone_names}")
        return None

    foot_pos_mesh = ref_pose.get_bone_pose(foot_bone, unreal.AnimPoseSpaces.WORLD).translation

    # mesh 本地前进方向：(right - left) 的 XY 分量顺时针转 90° 得到前向。
    # 无双脚时退回到 mesh 本地 +X。
    if left_bone and right_bone:
        lp = ref_pose.get_bone_pose(left_bone, unreal.AnimPoseSpaces.WORLD).translation
        rp = ref_pose.get_bone_pose(right_bone, unreal.AnimPoseSpaces.WORLD).translation
        rx, ry = rp.x - lp.x, rp.y - lp.y
        fx_mesh, fy_mesh = ry, -rx
        norm = math.hypot(fx_mesh, fy_mesh) or 1.0
        fx_mesh, fy_mesh = fx_mesh / norm, fy_mesh / norm
    else:
        fx_mesh, fy_mesh = 1.0, 0.0

    forward_offset_deg = math.degrees(math.atan2(fy_mesh, fx_mesh))
    foot_height_cm = float(foot_pos_mesh.z)

    result = {
        "forward_offset_deg": float(forward_offset_deg),
        "foot_height_cm": foot_height_cm,
        "foot_bone": foot_bone,
        "left_bone": left_bone,
        "right_bone": right_bone,
        "foot_pos_mesh": [float(foot_pos_mesh.x), float(foot_pos_mesh.y), float(foot_pos_mesh.z)],
        "forward_vec_mesh": [float(fx_mesh), float(fy_mesh), 0.0],
    }
    unreal.log(
        f"[skeleton] skmesh property: forward_offset={forward_offset_deg:+.2f}°, "
        f"foot_height={foot_height_cm:+.2f}cm (foot='{foot_bone}')"
    )
    return result


def estimate_blueprint_property(blueprint_asset):
    """估算人物蓝图在 actor 空间下的朝向偏角与脚部相对高度。

    从蓝图 CDO 取 SkeletalMeshComponent，委托 estimate_skeletalmesh_property 做
    skeleton-level 的估算；再用 mesh 组件的相对变换（location / rotation yaw /
    scale3d）把 mesh-local 的前进向量与脚部位置映射到蓝图 actor 空间。

    Returns dict（失败返回 None）:
        forward_offset_deg: 蓝图 +X 轴到人物正面朝前方向的夹角（度，绕 +Z，CCW 为正）。
        foot_height_cm:     脚部关节在 actor 空间中的 Z 坐标（cm）。
        foot_bone / left_bone / right_bone: 继承自 estimate_skeletalmesh_property。
        mesh_rel_loc / mesh_rel_rot:        SkeletalMeshComponent 相对 actor 根的变换。
        foot_pos_actor:                     脚骨在 actor 空间的 (x, y, z)。
        skeletal_mesh_property:             estimate_skeletalmesh_property 的原始返回。
    """
    if not isinstance(blueprint_asset, unreal.Blueprint):
        unreal.log_error("[skeleton] estimate_blueprint_property: not a Blueprint asset")
        return None

    cdo = unreal.get_default_object(blueprint_asset.generated_class())
    if not isinstance(cdo, unreal.Character):
        unreal.log_error("[skeleton] estimate_blueprint_property: CDO is not a Character")
        return None

    mesh_comp = cdo.mesh
    skeletal_mesh = getattr(mesh_comp, "skeletal_mesh", None) if mesh_comp else None
    if skeletal_mesh is None:
        unreal.log_error("[skeleton] estimate_blueprint_property: no skeletal mesh on CDO")
        return None

    sk_props = estimate_skeletalmesh_property(skeletal_mesh)
    if sk_props is None:
        return None

    rel_loc = mesh_comp.get_editor_property("relative_location")
    rel_rot = mesh_comp.get_editor_property("relative_rotation")
    rel_scale = mesh_comp.get_editor_property("relative_scale3d")

    # 仅用 yaw 把 mesh 本地方向 / 脚部位置映射到 actor 空间
    # （UE Character 惯例：mesh 组件的 roll/pitch ≈ 0；Z 分量不受 yaw 影响）
    yaw_rad = math.radians(rel_rot.yaw)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

    fx_mesh, fy_mesh, _ = sk_props["forward_vec_mesh"]
    fx_actor = cos_y * fx_mesh - sin_y * fy_mesh
    fy_actor = sin_y * fx_mesh + cos_y * fy_mesh
    forward_offset_deg = math.degrees(math.atan2(fy_actor, fx_actor))

    fpx, fpy, fpz = sk_props["foot_pos_mesh"]
    sx = fpx * rel_scale.x
    sy = fpy * rel_scale.y
    sz = fpz * rel_scale.z
    fax = cos_y * sx - sin_y * sy + rel_loc.x
    fay = sin_y * sx + cos_y * sy + rel_loc.y
    faz = sz + rel_loc.z
    foot_height_cm = rel_loc.z
    capsule_comp = cdo.get_component_by_class(unreal.CapsuleComponent)

    if capsule_comp:
        # 方式 A: 获取逻辑尺寸（半径和半高）
        radius = capsule_comp.get_scaled_capsule_radius()
        half_height = capsule_comp.get_scaled_capsule_half_height()
        
        print(f"Capsule Radius: {radius}")
        print(f"Capsule Half Height: {half_height}")

        foot_height_cm = half_height  # 以胶囊体底部为基准的脚高
    else:
        print("未在蓝图中找到胶囊体组件")

    result = {
        "forward_offset_deg": float(forward_offset_deg),
        "foot_height_cm": foot_height_cm,
        "half_height": half_height,
        "foot_bone": sk_props["foot_bone"],
        "left_bone": sk_props["left_bone"],
        "right_bone": sk_props["right_bone"],
        "mesh_rel_loc": [float(rel_loc.x), float(rel_loc.y), float(rel_loc.z)],
        "mesh_rel_rot": [float(rel_rot.roll), float(rel_rot.pitch), float(rel_rot.yaw)],
        "foot_pos_actor": [float(fax), float(fay), float(faz)],
        "skeletal_mesh_property": sk_props,
    }
    unreal.log(
        f"[skeleton] bp property: forward_offset={forward_offset_deg:+.2f}°, "
        f"foot_height={faz:+.2f}cm (foot='{sk_props['foot_bone']}'), "
        f"mesh_rel=loc({rel_loc.x:.1f},{rel_loc.y:.1f},{rel_loc.z:.1f}) "
        f"rot({rel_rot.roll:.1f},{rel_rot.pitch:.1f},{rel_rot.yaw:.1f})"
    )
    return result


def is_compatible_by_hierarchy(source_skeleton, target_skeleton,
                                check_pose=True, pos_tol=1e-3, rot_tol=1e-3, scale_tol=1e-3,
                                ignore_prefixes=("ik_", "hand_l_wep", "hand_r_wep", "root")):
    """
    判断两个骨架是否兼容（用于动画重定向）。
    判定标准：
      1. source 的核心骨骼在 target 中都存在（可忽略 IK 辅助骨骼和挂载点）
      2. check_pose=True 时，验证共有骨骼的 LOCAL/WORLD 参考姿势在容差内相等

    Args:
        ignore_prefixes: 忽略以这些前缀开头的骨骼（默认忽略 IK 辅助骨骼和武器挂载点）
    """
    if not source_skeleton or not target_skeleton:
        return False

    if source_skeleton == target_skeleton:
        return True

    source_pose = source_skeleton.get_reference_pose()
    target_pose = target_skeleton.get_reference_pose()

    source_bones_all = [str(b) for b in source_pose.get_bone_names()]
    target_bones = [str(b) for b in target_pose.get_bone_names()]

    # 过滤掉 IK 辅助骨骼和挂载点，只保留核心变形骨骼
    source_bones = [b for b in source_bones_all
                    if not any(b.startswith(p) or b == p for p in ignore_prefixes)]

    target_bone_set = set(target_bones)
    missing = [b for b in source_bones if b not in target_bone_set]
    if missing:
        print(f"source bones not all found in target: {missing}")
        return False

    if not check_pose:
        return True

    spaces_to_check = [unreal.AnimPoseSpaces.LOCAL, unreal.AnimPoseSpaces.WORLD]
    for space in spaces_to_check:
        for bone_name in source_bones:
            src_tf = source_pose.get_bone_pose(bone_name, space)
            tgt_tf = target_pose.get_bone_pose(bone_name, space)
            if not _transform_almost_equal(src_tf, tgt_tf, pos_tol, rot_tol, scale_tol):
                print("Distance exceed")
                return False

    return True


def _transform_almost_equal(a, b, pos_tol, rot_tol, scale_tol):
    """比较两个 unreal.Transform 是否在容差范围内相等。"""
    if (a.translation - b.translation).size() > pos_tol:
        return False

    ra, rb = a.rotation, b.rotation
    dot = ra.x * rb.x + ra.y * rb.y + ra.z * rb.z + ra.w * rb.w
    if 1.0 - abs(dot) > rot_tol:
        return False

    if (a.scale3d - b.scale3d).size() > scale_tol:
        return False

    return True


if __name__ == "__main__":
    # 测试用例：打印某个蓝图的估算属性
    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()
    skeletalmesh = "/Game/Population_System/Characters/Mans/child01_m/child01_m_highpoly"
    skeletalmesh = unreal.load_asset(skeletalmesh)
    print(estimate_skeletalmesh_property(skeletalmesh))