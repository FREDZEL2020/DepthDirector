import torch
import numpy as np
import torch.nn.functional as F
import open3d as o3d


# ================== from EX-4D ==========================
def point_to_mesh_cuda(pts, rgbs, faces, min_angle_deg=2.5):
    h, w, feature_channel= rgbs.shape
    vertices = pts.reshape(-1, 3)
    # colors = rgbs.reshape(-1, 3)
    masks = torch.ones((h, w, 1), dtype=torch.uint8).to(rgbs.device)
    rgbs = torch.cat([rgbs, masks], axis=-1)
    colors = rgbs.reshape(-1, feature_channel+1)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    face_normals = F.normalize(face_normals, dim=1)
    def angle_between(v1, v2):
        cos_theta = torch.sum(v1 * v2, -1) / (
            torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-12
        )
        return torch.arccos(torch.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi

    a = angle_between(v1 - v0, v2 - v0)
    b = angle_between(v2 - v1, v0 - v1)
    c = angle_between(v0 - v2, v1 - v2)
    min_angles = torch.minimum(torch.minimum(a, b), c)

    # Filter faces based on minimum angle
    valid_faces = min_angles >= min_angle_deg
    z_range = vertices[:, 2].max() - vertices[:, 2].min()

    z01, z12, z20 = torch.abs((v0 - v1)[:, 2]), torch.abs((v1 - v2)[:, 2]), torch.abs((v2 - v0)[:, 2])
    y01, y12, y20 = torch.abs((v0 - v1)[:, 1]), torch.abs((v1 - v2)[:, 1]), torch.abs((v2 - v0)[:, 1])
    x01, x12, x20 = torch.abs((v0 - v1)[:, 0]), torch.abs((v1 - v2)[:, 0]), torch.abs((v2 - v0)[:, 0])
    z_max = torch.maximum(torch.maximum(z01, z12), z20)
    y_max = torch.maximum(torch.maximum(y01, y12), y20)
    x_max = torch.maximum(torch.maximum(x01, x12), x20)
    proj_max = torch.maximum(torch.maximum(x_max, y_max), z_max)
    valid_faces2 = (proj_max / z_range < 0.0013)
    valid_faces_final = valid_faces & valid_faces2
    invalid_faces = faces[~valid_faces]
    colors.index_put_((invalid_faces,), torch.zeros(feature_channel+1).to(colors.device))
    return vertices, faces, colors, valid_faces

def compute_vertex_normals(vertices, faces):
    """
    从网格计算顶点法向量
    
    参数:
    - vertices: 顶点坐标 [N, 3]
    - faces: 面索引 [M, 3]
    
    返回:
    - vertex_normals: 顶点法向量 [N, 3]
    """
    vertices = torch.as_tensor(vertices)
    faces = torch.as_tensor(faces)
    
    # 计算面法向量
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]] 
    v2 = vertices[faces[:, 2]]
    
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_normals = torch.nn.functional.normalize(face_normals, p=2, dim=1)
    
    # 初始化顶点法向量
    vertex_normals = torch.zeros_like(vertices)
    
    # 累加面法向量到顶点
    vertex_normals = vertex_normals.index_add(0, faces[:, 0], face_normals)
    vertex_normals = vertex_normals.index_add(0, faces[:, 1], face_normals) 
    vertex_normals = vertex_normals.index_add(0, faces[:, 2], face_normals)
    
    # 归一化
    vertex_normals = torch.nn.functional.normalize(vertex_normals, p=2, dim=1)
    
    return vertex_normals

def render_nvdiffrast(glctx, vertices, faces, colors, proj, poses, fovx, fovy, h, w, scale=1.0, near=1.0, far=50):
    import nvdiffrast.torch as dr
    # x right y down z forward
    def transform_pos(mtx, pos):
        t_mtx = torch.from_numpy(mtx).to(pos.device) if isinstance(mtx, np.ndarray) else mtx
        # (x,y,z) -> (x,y,z,1)
        posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos.device)], axis=1)
        return torch.matmul(posw, t_mtx.t())[None, ...]

    def render(glctx, mtx, pos, pos_idx, vtx_col, col_idx, h, w):
        pos_clip    = transform_pos(mtx, pos)
        rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[h, w])
        visible_faces = rast_out[...,3][rast_out[...,3] > 0] - 1
        depth, _    = dr.interpolate(pos_clip, rast_out, col_idx)
        depth = depth[...,-2]
        color, _    = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
        color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
        
        
        return color, depth
    # poses[0,:] *= -1
    # poses[1,:] *= -1
    # poses[2,:] *= -1
    mvp = proj @ poses
    return render(glctx, mvp, vertices, faces, colors, faces, h, w)


def render_shaded_mesh(glctx, vertices, faces, normals, proj, poses, light_direction=None, 
                      h=512, w=512, ambient_intensity=0.3, diffuse_intensity=0.7):
    import nvdiffrast.torch as dr
    """
    渲染阴影着色的mesh
    
    参数:
    - glctx: nvdiffrast上下文
    - vertices: 顶点坐标 [N, 3]
    - faces: 面索引 [M, 3]
    - normals: 顶点法向量 [N, 3]
    - proj: 投影矩阵 [4, 4]
    - poses: 视图矩阵 [4, 4]
    - light_direction: 光源方向 [3,]，如果为None则使用默认方向
    - h, w: 输出图像的高度和宽度
    - ambient_intensity: 环境光强度
    - diffuse_intensity: 漫反射强度
    """
    
    def transform_pos(mtx, pos):
        t_mtx = torch.from_numpy(mtx).to(pos.device) if isinstance(mtx, np.ndarray) else mtx
        posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos.device)], axis=1)
        return torch.matmul(posw, t_mtx.t())[None, ...]
    
    def render_lambertian(glctx, mtx, pos, pos_idx, vtx_normals, normal_idx, light_dir, h, w):
        # 变换顶点位置到裁剪空间
        pos_clip = transform_pos(mtx, pos)
        
        # 光栅化
        rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[h, w])
        
        # 插值法向量
        normals_interpolated, _ = dr.interpolate(vtx_normals[None, ...], rast_out, normal_idx)
        
        # 归一化插值后的法向量
        normals_interpolated = torch.nn.functional.normalize(normals_interpolated, p=2, dim=-1)
        
        # 计算朗伯反射 (漫反射)
        # 将光源方向扩展到与法向量相同的形状
        light_dir = light_dir.view(1, 1, 1, 3).expand_as(normals_interpolated)
        light_dir = torch.nn.functional.normalize(light_dir, p=2, dim=-1)
        
        # 计算法向量和光源方向的点积 (朗伯余弦定律)
        # 使用clamp确保值在[0,1]范围内
        lambert = torch.clamp(torch.sum(normals_interpolated * light_dir, dim=-1, keepdim=True), 0, 1)
        
        # 应用环境光和漫反射
        shaded = ambient_intensity + diffuse_intensity * lambert
        
        # 抗锯齿
        shaded = dr.antialias(shaded, rast_out, pos_clip, pos_idx)
        
        return shaded[0]  # 移除batch维度
    
    # 设置默认光源方向 (从右上方前方照射)
    if light_direction is None:
        light_direction = torch.tensor([0.5, 0.5, 1.0], device=vertices.device, dtype=torch.float32)
    else:
        light_direction = torch.tensor(light_direction, device=vertices.device, dtype=torch.float32)
    
    # 归一化光源方向
    light_direction = torch.nn.functional.normalize(light_direction, p=2, dim=0)
    
    # 计算MVP矩阵
    mvp = proj @ poses
    
    # 渲染阴影图像
    shaded_image = render_lambertian(glctx, mvp, vertices, faces, normals, faces, light_direction, h, w)
    
    return shaded_image


def generate_faces(H, W, idx, device, left_padding=0, right_padding=0, top_padding=0, bottom_padding=0):
    idx = np.arange(H * W).reshape(H, W)
    idx = idx[top_padding:H-bottom_padding, left_padding:W-right_padding]
    faces = torch.from_numpy(np.concatenate([
            np.stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[:-1, 1:].ravel()], axis=-1),
            np.stack([idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()], axis=-1)
        ], axis=0)).int().to(device)
    return faces

def getprojection(fovx, fovy, n=1.0, f=50.0, device=None):
    x = np.tan(fovx * 0.5) * n
    y = np.tan(fovy * 0.5) * n
    return torch.tensor([[n/x,    0,            0,              0],
                         [  0, n/-y,            0,              0],
                         [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                         [  0,    0,           -1,              0]], dtype=torch.float32, device=device)


def get_rays(directions, c2w):     # directions: (H*W, 3)
    # c2w: (4, 4)
    # return: rays_o (H*W, 3), rays_d (H*W, 3)
    
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = torch.einsum('kj,ij->ik', c2w[:3, :3], directions)  # (H*W, 3)
    
    # Normalize ray directions
    # rays_d = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-12)
    
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].unsqueeze(0).expand(directions.shape[0], -1)  # (H*W, 3)
    
    return rays_o, rays_d

def get_rays_from_pose(pose, K, H, W):
    # pose: (4, 4)
    # K: (3, 3)
    # H, W: int
    # return: (H, W, 3), (H, W, 3)
    
    # Create a meshgrid for screen coordinates
    rays_screen_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack(rays_screen_coords, dim=-1).reshape(H * W, 2).to(pose)
    i, j = grid[..., 1], grid[..., 0]
    
    # Extract intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Calculate directions
    directions = torch.stack([
        (i - cx) / fx,
        (j - cy) / fy,
        torch.ones_like(i)
    ], dim=-1)  # (H*W, 3)
    
    # Get rays
    ro, rd = get_rays(directions, pose)  # (H*W, 3), (H*W, 3)
    
    # Reshape rays
    ro = ro.reshape(H, W, 3)
    rd = rd.reshape(H, W, 3)
    
    return ro, rd
