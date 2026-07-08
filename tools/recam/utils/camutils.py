

import math
import copy
import torch
import numpy as np

def sphere2pose(c2ws_input, theta, phi, r, device, x=None, y=None):
    c2ws = copy.deepcopy(c2ws_input)
    # c2ws[:,2, 3] = c2ws[:,2, 3] - radius

    # 先沿着世界坐标系z轴方向平移再旋转
    c2ws[:, 2, 3] -= r
    if x is not None:
        c2ws[:, 1, 3] += y
    if y is not None:
        c2ws[:, 0, 3] -= x

    theta = torch.deg2rad(torch.tensor(theta)).to(device)
    sin_value_x = torch.sin(theta)
    cos_value_x = torch.cos(theta)
    rot_mat_x = (
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, cos_value_x, -sin_value_x, 0],
                [0, sin_value_x, cos_value_x, 0],
                [0, 0, 0, 1],
            ]
        )
        .unsqueeze(0)
        .repeat(c2ws.shape[0], 1, 1)
        .to(device)
    )

    phi = torch.deg2rad(torch.tensor(phi)).to(device)
    sin_value_y = torch.sin(phi)
    cos_value_y = torch.cos(phi)
    rot_mat_y = (
        torch.tensor(
            [
                [cos_value_y, 0, sin_value_y, 0],
                [0, 1, 0, 0],
                [-sin_value_y, 0, cos_value_y, 0],
                [0, 0, 0, 1],
            ]
        )
        .unsqueeze(0)
        .repeat(c2ws.shape[0], 1, 1)
        .to(device)
    )

    c2ws = torch.matmul(rot_mat_x, c2ws)
    c2ws = torch.matmul(rot_mat_y, c2ws)
    # c2ws[:,2, 3] = c2ws[:,2, 3] + radius
    return c2ws


def generate_traj_specified(c2ws_anchor, theta, phi, d_r, d_x, d_y, frame, device):
    # Initialize a camera.
    thetas = np.linspace(0, theta, frame)
    phis = np.linspace(0, phi, frame)
    rs = np.linspace(0, d_r, frame)
    xs = np.linspace(0, d_x, frame)
    ys = np.linspace(0, d_y, frame)
    c2ws_list = []
    for th, ph, r, x, y in zip(thetas, phis, rs, xs, ys):
        c2w_new = sphere2pose(
            c2ws_anchor,
            np.float32(th),
            np.float32(ph),
            np.float32(r),
            device,
            np.float32(x),
            np.float32(y),
        )
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list, dim=0)
    return c2ws

from scipy.interpolate import UnivariateSpline, interp1d
def txt_interpolation(input_list, n, mode='smooth'):
    x = np.linspace(0, 1, len(input_list))
    if mode == 'smooth':
        f = UnivariateSpline(x, input_list, k=3)
    elif mode == 'linear':
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)
    return ynew


def generate_traj_txt(c2ws_anchor, phi, theta, r, frame, device):
    # Initialize a camera.
    """
    The camera coordinate sysmte in COLMAP is right-down-forward
    Pytorch3D is left-up-forward
    """

    if len(phi) > 3:
        phis = txt_interpolation(phi, frame, mode='smooth')
        phis[0] = phi[0]
        phis[-1] = phi[-1]
    else:
        phis = txt_interpolation(phi, frame, mode='linear')

    if len(theta) > 3:
        thetas = txt_interpolation(theta, frame, mode='smooth')
        thetas[0] = theta[0]
        thetas[-1] = theta[-1]
    else:
        thetas = txt_interpolation(theta, frame, mode='linear')

    if len(r) > 3:
        rs = txt_interpolation(r, frame, mode='smooth')
        rs[0] = r[0]
        rs[-1] = r[-1]
    else:
        rs = txt_interpolation(r, frame, mode='linear')
    # rs = rs*c2ws_anchor[0,2,3].cpu().numpy()

    c2ws_list = []
    for th, ph, r in zip(thetas, phis, rs):
        c2w_new = sphere2pose(
            c2ws_anchor, np.float32(th), np.float32(ph), np.float32(r), device
        )
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list, dim=0)
    return c2ws

def euler_to_rotation_matrix(yaw, pitch, roll):
    """
    将欧拉角（pitch, yaw, roll）转换为旋转矩阵
    旋转顺序：Yaw (Z) -> Pitch (X) -> Roll (Y)
    """
    # 转换为弧度
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    roll_rad = np.radians(roll)
    
    # UE旋转顺序：Yaw(Z) -> Pitch(Y) -> Roll(X)
    R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]])
    
    R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]])
    
    R_ue = R_z @ R_y @ R_x   # 组合旋转顺序
    return R_ue


def location_rotation_to_transform_matrix(location, rotation):
    """
    将位置和欧拉角转换为4x4变换矩阵
    """
    # 提取位置和旋转
    x, y, z = location
    pitch, yaw, roll = rotation['pitch'], rotation['yaw'], rotation['roll']
    # yaw = yaw
    R_ue = euler_to_rotation_matrix(yaw, -pitch, roll)
    T_ue2cv = np.array([[0, 1, 0],
                        [0, 0, -1],
                        [1, 0, 0]])
    T_lr = np.array([[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 1]
                    ])
    
    # OpenCV旋转矩阵
    R_cv = T_lr @ R_ue @ np.linalg.inv(T_ue2cv)
    t_cv = np.array([y, x ,z])

    # R_cv = euler_angles_to_matrix(np.array([-yaw, -pitch, roll]), ["Y","X","Z"])
    # 构建4x4变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R_cv
    transform_matrix[:3, 3] = t_cv
    return transform_matrix

