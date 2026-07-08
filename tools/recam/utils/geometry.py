# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


# FROM Vipe
def align_inv_depth_to_depth(
    source_inv_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    quantile_masking: bool = True,
) -> tuple[torch.Tensor, float, float]:
    """
    Apply affine transformation to align source inverse depth to target depth.

    Args:
        source_inv_depth: Inverse depth map to be aligned. Shape: (H, W).
        target_depth: Target depth map. Shape: (H, W).
        target_mask: Mask of valid target pixels. Shape: (H, W).

    Returns:
        Aligned Depth map. Shape: (H, W).
        scale: Scaling factor.
        bias: Bias term.
    """
    source_inv_depth_original = source_inv_depth.clone()
    step = 4
    source_inv_depth = source_inv_depth[...,::step]
    target_depth = target_depth[...,::step]
    target_mask = target_mask[...,::step]
    target_inv_depth = 1.0 / target_depth
    source_mask = source_inv_depth > 0
    target_depth_mask = target_depth > 0

    if target_mask is None:
        target_mask = target_depth_mask
    else:
        target_mask = torch.logical_and(target_mask > 0, target_depth_mask)

    # Remove outliers
    if quantile_masking:
        outlier_quantiles = torch.tensor([0.1, 0.9], device=source_inv_depth.device)
        source_data_low, source_data_high = torch.quantile(source_inv_depth[source_mask], outlier_quantiles)
        target_data_low, target_data_high = torch.quantile(target_inv_depth[target_mask], outlier_quantiles)
        source_mask = (source_inv_depth > source_data_low) & (source_inv_depth < source_data_high)
        target_mask = (target_inv_depth > target_data_low) & (target_inv_depth < target_data_high)

    mask = torch.logical_and(source_mask, target_mask)

    source_data = source_inv_depth[mask].view(-1, 1)
    target_data = target_inv_depth[mask].view(-1, 1)

    ones = torch.ones((source_data.shape[0], 1), device=source_data.device)
    source_data_h = torch.cat([source_data, ones], dim=1)

    transform_matrix = torch.linalg.lstsq(source_data_h, target_data).solution

    scale, bias = transform_matrix[0, 0], transform_matrix[1, 0]
    aligned_inv_depth = source_inv_depth_original * scale + bias
    aligned_depth = torch.clamp(aligned_inv_depth.reciprocal(), min=1e-4)

    return aligned_depth, scale, bias


def align_depth_to_depth(
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    quantile_masking: bool = True,
    bias: bool = True,
) -> torch.Tensor:
    """
    Apply affine transformation to align source depth to target depth.

    Args:
        source_inv_depth: Depth map to be aligned. Shape: (H, W).
        target_depth: Target depth map. Shape: (H, W).
        target_mask: Mask of valid target pixels. Shape: (H, W).

    Returns:
        Aligned Depth map. Shape: (H, W).
    """
    source_mask = source_depth > 0
    target_depth_mask = target_depth > 0

    if target_mask is None:
        target_mask = target_depth_mask
    else:
        target_mask = torch.logical_and(target_mask > 0, target_depth_mask)

    # Remove outliers
    if quantile_masking:
        outlier_quantiles = torch.tensor([0.1, 0.9], device=source_depth.device)

        source_data_low, source_data_high = torch.quantile(source_depth[source_mask], outlier_quantiles)
        target_data_low, target_data_high = torch.quantile(target_depth[target_mask], outlier_quantiles)
        source_mask = (source_depth > source_data_low) & (source_depth < source_data_high)
        target_mask = (target_depth > target_data_low) & (target_depth < target_data_high)

    mask = torch.logical_and(source_mask, target_mask)

    source_data = source_depth[mask].view(-1, 1)
    target_data = target_depth[mask].view(-1, 1)

    if not bias:
        ones = torch.ones((source_data.shape[0], 1), device=source_data.device)
        source_data_h = torch.cat([source_data, ones], dim=1)
        transform_matrix = torch.linalg.lstsq(source_data_h, target_data).solution

        scale, bias = transform_matrix[0, 0], transform_matrix[1, 0]
        aligned_depth = source_depth * scale + bias
        aligned_depth = torch.clamp(aligned_depth, min=1e-4)

    else:
        scale = torch.median(target_data / source_data)
        aligned_depth = source_depth * scale
        aligned_depth = torch.clamp(aligned_depth, min=1e-4)

    return aligned_depth


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray
) -> np.ndarray:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (np.ndarray): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)

    Returns:
        np.ndarray: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]
        )
        world_points_list.append(cur_world_points)
    world_points_array = np.stack(world_points_list, axis=0)

    return world_points_array


def unproject_depth_map_to_point_map_torch(
    depth_map: torch.Tensor, extrinsics_cam: torch.Tensor, intrinsics_cam: torch.Tensor
) -> torch.Tensor:
    """
    Unproject a batch of depth maps to 3D world coordinates (PyTorch version).

    Args:
        depth_map (torch.Tensor): Batch of depth maps of shape (S, H, W) or (S, H, W, 1)
        extrinsics_cam (torch.Tensor): Batch of camera extrinsic matrices of shape (S, 3, 4), cam-from-world
        intrinsics_cam (torch.Tensor): Batch of camera intrinsic matrices of shape (S, 3, 3)

    Returns:
        torch.Tensor: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    if depth_map.dim() == 4:
        depth_map = depth_map.squeeze(-1)  # (S, H, W)

    S, H, W = depth_map.shape
    device = depth_map.device
    dtype = depth_map.dtype

    # Intrinsic parameters: (S,)
    fu = intrinsics_cam[:, 0, 0]
    fv = intrinsics_cam[:, 1, 1]
    cu = intrinsics_cam[:, 0, 2]
    cv = intrinsics_cam[:, 1, 2]

    # Pixel grid: (H, W)
    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    # Expand to (S, H, W)
    u_grid = u_grid.unsqueeze(0).expand(S, -1, -1)
    v_grid = v_grid.unsqueeze(0).expand(S, -1, -1)

    # Unproject to camera coords: (S, H, W)
    x_cam = (u_grid - cu[:, None, None]) * depth_map / fu[:, None, None]
    y_cam = (v_grid - cv[:, None, None]) * depth_map / fv[:, None, None]
    z_cam = depth_map

    # Camera coords: (S, H, W, 3)
    cam_coords = torch.stack([x_cam, y_cam, z_cam], dim=-1)

    # Invert extrinsics (cam-from-world -> world-from-cam)
    # extrinsics_cam: (S, 3, 4) -> pad to (S, 4, 4) for closed_form_inverse_se3
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)
    bottom = bottom.view(1, 1, 4).expand(S, -1, -1)  # (S, 1, 4)
    extrinsics_4x4 = torch.cat([extrinsics_cam, bottom], dim=1)  # (S, 4, 4)

    cam_to_world = closed_form_inverse_se3(extrinsics_4x4)  # (S, 4, 4)
    R = cam_to_world[:, :3, :3]  # (S, 3, 3)
    t = cam_to_world[:, :3, 3]   # (S, 3)

    # Transform to world coords: (S, H, W, 3)
    # einsum: S,H,W,3 x S,3,3 -> S,H,W,3
    world_coords = torch.einsum("shwj,sij->shwi", cam_coords, R) + t[:, None, None, :]

    return world_coords


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)
    # cam_coords = np.stack((z_cam, x_cam, -y_cam), axis=-1).astype(np.float32)
    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


# TODO: this code can be further cleaned up


def project_world_points_to_camera_points_batch(world_points, cam_extrinsics):
    """
    Transforms 3D points to 2D using extrinsic and intrinsic parameters.
    Args:
        world_points (torch.Tensor): 3D points of shape BxSxHxWx3.
        cam_extrinsics (torch.Tensor): Extrinsic parameters of shape BxSx3x4.
    Returns:
    """
    # TODO: merge this into project_world_points_to_cam
    
    # device = world_points.device
    # with torch.autocast(device_type=device.type, enabled=False):
    ones = torch.ones_like(world_points[..., :1])  # shape: (B, S, H, W, 1)
    world_points_h = torch.cat([world_points, ones], dim=-1)  # shape: (B, S, H, W, 4)

    # extrinsics: (B, S, 3, 4) -> (B, S, 1, 1, 3, 4)
    extrinsics_exp = cam_extrinsics.unsqueeze(2).unsqueeze(3)

    # world_points_h: (B, S, H, W, 4) -> (B, S, H, W, 4, 1)
    world_points_h_exp = world_points_h.unsqueeze(-1)

    # Now perform the matrix multiplication
    # (B, S, 1, 1, 3, 4) @ (B, S, H, W, 4, 1) broadcasts to (B, S, H, W, 3, 1)
    camera_points = torch.matmul(extrinsics_exp, world_points_h_exp).squeeze(-1)

    return camera_points



def project_world_points_to_cam(
    world_points,
    cam_extrinsics,
    cam_intrinsics=None,
    distortion_params=None,
    default=0,
    only_points_cam=False,
):
    """
    Transforms 3D points to 2D using extrinsic and intrinsic parameters.
    Args:
        world_points (torch.Tensor): 3D points of shape Px3.
        cam_extrinsics (torch.Tensor): Extrinsic parameters of shape Bx3x4.
        cam_intrinsics (torch.Tensor): Intrinsic parameters of shape Bx3x3.
        distortion_params (torch.Tensor): Extra parameters of shape BxN, which is used for radial distortion.
    Returns:
        torch.Tensor: Transformed 2D points of shape BxNx2.
    """
    device = world_points.device
    # with torch.autocast(device_type=device.type, dtype=torch.double):
    with torch.autocast(device_type=device.type, enabled=False):
        N = world_points.shape[0]  # Number of points
        B = cam_extrinsics.shape[0]  # Batch size, i.e., number of cameras
        world_points_homogeneous = torch.cat(
            [world_points, torch.ones_like(world_points[..., 0:1])], dim=1
        )  # Nx4
        # Reshape for batch processing
        world_points_homogeneous = world_points_homogeneous.unsqueeze(0).expand(
            B, -1, -1
        )  # BxNx4

        # Step 1: Apply extrinsic parameters
        # Transform 3D points to camera coordinate system for all cameras
        cam_points = torch.bmm(
            cam_extrinsics, world_points_homogeneous.transpose(-1, -2)
        )

        if only_points_cam:
            return None, cam_points

        # Step 2: Apply intrinsic parameters and (optional) distortion
        image_points = img_from_cam(cam_intrinsics, cam_points, distortion_params, default=default)

        return image_points, cam_points

