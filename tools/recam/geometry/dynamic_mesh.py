
import torch
import numpy as np
import math
import trimesh
from tqdm import tqdm
from utils.dwmesh import *
try:
    import nvdiffrast.torch as dr
    glctx = dr.RasterizeCudaContext(device='cuda:0')
except Exception as e:
    print(e)

class Mesh():
    def __init__(self, device):
        self.device = device
        

    def generate_mesh(self, rgb_map, depth_map, K, pose):
        """
        Args:
            rgb_map: torch.tensor H,W,3
            depth_map: torch.tensor H,W,1
            K: 3,3
            pose: 4,4
        """
        H, W = depth_map.shape[:2]
        device = self.device
        fidx = np.arange(H * W).reshape(H, W)
        faces = generate_faces(H, W, fidx, device)

        border = 1000 # (depth_src.max() // 100) * 100
        depth_map = depth_map.clip(0,border-100)
        depth_map[0, :] = border
        depth_map[-1, :] = border
        depth_map[:, 0] = border
        depth_map[:, -1] = border
        
        fov_y = 2 * math.atan2(K[1, 2] * 2, 2 * K[1, 1])
        fov_x = 2 * math.atan2(K[0, 2] * 2, 2 * K[0, 0])
        # Create a camera intrinsic matrix, convert fov_deg to focal length
        fx = fy = 0.5 * H / math.tan(fov_y / 2)
        Ks = torch.tensor([[fx, 0, W / 2],
                            [0, fy, H / 2],
                            [0, 0, 1]], dtype=torch.float32, device=device)
        # w2c -> c2w     
        cam_s = torch.linalg.inv(pose)

        ro_src, rd_src = get_rays_from_pose(cam_s, Ks, H, W)  # (H, W, 3)  
        pts_xyz = ro_src + rd_src * depth_map  # H,W,3
        vertices, dw_faces, colors, valid_faces = point_to_mesh_cuda(pts_xyz, rgb_map, faces)
        
        colors = colors.to(self.device)
        return vertices, dw_faces, colors, valid_faces
        
        
class Dynamic3DMesh():
    def __init__(self, device):
        self.device = device
        self.mesh = Mesh(device)
    
    def build(self, frames, depths, Ks, poses):
        self.frames = frames
        self.depths = depths
        self.Ks = Ks
        self.poses = poses

        # build when render

    @torch.no_grad()
    def render(self, poses_t, Ks, base_idx=0):
        """_summary_

        Args:
            frames (_type_): L,H,W,D, D-Dimension feature(colors)
            depths (_type_): L,H,W,1
            Ks (_type_): _description_
            poses_s (_type_): _description_
            poses_t (_type_): _description_
            t_frames (_type_, optional): _description_. Defaults to None.
            save_mesh (bool, optional): _description_. Defaults to False.
        Returns:
            warp: L,H,W,D
            depth: L,H,W,3
        """
        device = self.device
        # L,H,W,C
        global glctx
            
        video = []
        depth_video = []
        dw_depth_video = []
        dw_video = []
        meshes = {}
        K = Ks[0]
        for idx in tqdm(range(len(poses_t)), desc="DWMesh"):
        # for idx in range(len(poses_t)):
            pose_s = torch.asarray(self.poses[idx], dtype=torch.float32, device=device)
            pose_t = torch.asarray(poses_t[idx], dtype=torch.float32, device=device)
            cam_s = torch.linalg.inv(pose_s)            
            cam_t = torch.linalg.inv(pose_t)
            cam_t = cam_t @ torch.asarray(np.diag([1,-1,-1,1])).to(cam_s.device).to(cam_s.dtype)
            fov_y = 2 * math.atan2(K[1, 2] * 2, 2 * K[1, 1])
            fov_x = 2 * math.atan2(K[0, 2] * 2, 2 * K[0, 0])
            proj = getprojection(fov_x, fov_y, n=1e-3, f=1e3, device=device)
            rgb_map = torch.asarray(self.frames[idx], dtype=torch.float32, device=device)
            H,W,C = rgb_map.shape

            depth_map = torch.asarray(self.depths[idx], dtype=torch.float32, device=device)  # (H, W, 3)
            vertices, dw_faces, colors, valid_faces = self.mesh.generate_mesh(rgb_map, depth_map, self.Ks[0], pose_s)
            # disable Depth watertight
            part_faces = dw_faces[valid_faces]
            img, depth = render_nvdiffrast(glctx, vertices, part_faces, colors, proj, torch.linalg.inv(cam_t), fov_x, fov_y, H, W)
            
            img = img[0]
            depth_valid = depth[0]
            if (pose_s-pose_t).abs().sum() < 1e-1 and idx+base_idx == 0:
                img[..., :C] = rgb_map
                img[..., C:] = 1
            else:
                mask = img[..., C:]
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                img[..., C:] = mask
                img[..., :C] = img[..., :C] * mask
            video.append(img.cpu().numpy())
            depth_video.append(depth.squeeze().cpu().numpy())

            # ======================== Render Pass 2 ==============================
            colors = colors.to(self.device)
            img, depth = render_nvdiffrast(glctx, vertices, dw_faces, colors, proj, torch.linalg.inv(cam_t), fov_x, fov_y, H, W)
            img = img[0]
            if (pose_s-pose_t).abs().sum() < 1e-4 and idx+base_idx == 0:
                img[..., :C] = rgb_map
                img[..., C:] = 1
            else:
                mask = img[..., C:]
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                img[..., C:] = mask
                img[..., :C] = img[..., :C] * (mask)
            dw_video.append(img.cpu().numpy())
            dw_depth_video.append(depth.squeeze().cpu().numpy())

            if idx+base_idx in [0,80] :
                # Convert RGBA colors to RGBA uint8 format that trimesh expects
                vertex_colors = (colors * 255).cpu().numpy().astype(np.uint8)[:, :3]

                # 先创建完整信息的 mesh
                full_mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), 
                                            faces=dw_faces[valid_faces].cpu().numpy(),
                                            vertex_colors=(colors * 255).cpu().numpy().astype(np.uint8)[:, :3])

                # 然后一键清理：它会删掉没用的顶点，并自动保留对应的颜色
                full_mesh.remove_unreferenced_vertices()

                dw_mesh = trimesh.Trimesh(vertices.cpu().numpy(), dw_faces.cpu().numpy())
                dw_mesh.visual.vertex_colors = vertex_colors
                invalid_mesh =  trimesh.Trimesh(vertices.cpu().numpy(), dw_faces[~valid_faces].cpu().numpy())
                invalid_mesh.remove_unreferenced_vertices()
                meshes[idx] ={
                    "valid": full_mesh,
                    "dw": dw_mesh,
                    "invalid":invalid_mesh
                }
                
                            
            # ================ get depth range
            if idx == 0:
                valid_depth_v = depth_valid[100:-100,100:-100][depth_valid[100:-100,100:-100] > 0]
                if valid_depth_v.shape[0] > 0:
                    depth_min = valid_depth_v.min().cpu().numpy().item()
                    depth_max = valid_depth_v.max().cpu().numpy().item()
                else:
                    depth_min = 0.1
                    depth_max = 100
                # print("depth min max", depth_min, depth_max)
            else:
                valid_depth_v = depth_valid[100:-100,100:-100][depth_valid[100:-100,100:-100] > 0]
                if valid_depth_v.shape[0] > 0:
                    depth_min = min(depth_min, torch.min(valid_depth_v).cpu().numpy().item())
                    depth_max = max(depth_max, torch.max(valid_depth_v).cpu().numpy().item())
            
            

        # print("depth min max", depth_min, depth_max)

        return {
            "warp": np.stack(video, axis=0)[...,:C],
            "dw_warp": np.stack(dw_video, axis=0)[...,:C],
            "depth": np.stack(depth_video, axis=0),
            "dw_depth": np.stack(dw_depth_video, axis=0),
            "mask": np.stack(video, axis=0)[...,C:],
            "dw_mask": np.stack(dw_video, axis=0)[...,C:],
            "extra":{
                "depth_min": depth_min,
                "depth_max": depth_max,
                "meshes": meshes
            }
        }
            