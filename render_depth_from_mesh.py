import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
from PIL import Image
import cv2
import pyrender
import trimesh
import os
from tools.mesh_renderer import Renderer
from tools import fusers_helper
from tools.mesh_renderer import (DEFAULT_CAM_FRUSTUM_MATERIAL,
                                 DEFAULT_MESH_MATERIAL, Renderer,
                                 SmoothBirdsEyeCamera, camera_marker,
                                 create_light_array, get_image_box,
                                 transform_trimesh)
def create_camera_frustum(scale=1.0):
    points = np.array([
        [0, 0, 0],
        [-0.5, -0.5, 1],
        [0.5, -0.5, 1],
        [0.5, 0.5, 1],
        [-0.5, 0.5, 1]
    ]) * scale
    lines = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ])
    colors = np.array([[1, 0, 0] for _ in range(len(lines))])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

rgb_path = "/data/laiyan/datasets/ScanNet/extracted/scans/scene0000_00/sensor_data/frame-000995.color.512.png"
depth_path = "/data/laiyan/datasets/ScanNet/extracted/scans/scene0000_00/sensor_data/frame-000995.depth.256.png"
pose_path = "/data/laiyan/datasets/ScanNet/extracted/scans/scene0000_00/sensor_data/frame-000995.pose.txt"
mesh_path = "/data/laiyan/datasets/ScanNet/extracted/scans/scene0000_00/scene0000_00_vh_clean_2.ply"

rgb = Image.open(rgb_path)
rgb = np.array(rgb)
rgb_height, rgb_width = rgb.shape[:2]
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.
depth_height, depth_width = depth.shape[:2]
world_T_cam = np.genfromtxt(pose_path).astype(np.float32)
cam_T_world = np.linalg.inv(world_T_cam)



# Create camera frustum
camera_frustum = create_camera_frustum(scale=0.3)
# Transform camera frustum according to the pose
camera_frustum.rotate(world_T_cam[:3,:3], center=(0,0,0))
camera_frustum.translate(world_T_cam[:3,3])

rgb_camera_intrinsics_path = "/data/laiyan/datasets/ScanNet/extracted/scans/scene0000_00/intrinsic/intrinsic_color.txt"
depth_camera_intrinsics_path = "/data/laiyan/datasets/ScanNet/extracted/scans/scene0000_00/intrinsic/intrinsic_depth.txt"
K_rgb = np.genfromtxt(rgb_camera_intrinsics_path)
K_depth = np.genfromtxt(depth_camera_intrinsics_path)
K_rgb = torch.tensor(K_rgb.astype(np.float32))
K_depth = torch.tensor(K_depth.astype(np.float32))

metadata_filename = "/data/laiyan/datasets/ScanNet/extracted/scans/scene0000_00/scene0000_00.txt"
def readlines(filepath):
    """Reads in a text file and returns lines in a list."""
    with open(filepath, "r") as f:
        lines = f.read().splitlines()
    return lines

# load in basic intrinsics for the full size depth map.
lines = readlines(metadata_filename)
lines = [line.split(" = ") for line in lines]
data = {key: val for key, val in lines}

# scale intrinsics to the dataset's configured depth resolution.
K_depth[0] *= depth_width / float(data["depthWidth"])
K_depth[1] *= depth_height / float(data["depthHeight"])
render_depth_width = depth_width
render_depth_height = depth_height
render_rgb_width = rgb_width
render_rgb_height = rgb_height

world_T_cam_44 = world_T_cam
K_33 = K_depth

fpv_renderer = Renderer(height=render_depth_height, width=render_depth_width)
light_pos = world_T_cam_44.copy()
light_pos[2, 3] += 5.0
lights = create_light_array(
    pyrender.PointLight(intensity=30.0),
    light_pos,
    x_length=12,
    y_length=12,
    num_x=6,
    num_y=6,
)
mesh = trimesh.load(mesh_path,force="mesh")
meshes = [mesh]

render_fpv_depth = fpv_renderer.render_mesh(
    meshes,
    render_depth_height, render_depth_width,
    world_T_cam_44, K_33,
    False,
)
render_fpv_color = fpv_renderer.render_mesh(
    meshes,
    render_depth_height, render_depth_width,
    world_T_cam_44, K_33,
    True,
)


plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(render_fpv_depth, cmap='jet')
plt.title('Rendered Depth')
plt.colorbar(label='Depth')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(depth, cmap='jet')
plt.title('Original Depth')
plt.colorbar(label='Depth')
plt.axis('off')



diff = np.abs(render_fpv_depth - depth)
plt.subplot(1, 3, 3)
plt.imshow(diff, cmap='jet',vmin=0,vmax=0.2)
plt.title('diff')
plt.colorbar(label='diff')
plt.axis('off')


plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(render_fpv_color)
plt.title('Rendered rgb')
plt.colorbar(label='rgb')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(rgb)
plt.title('Original rgb')
plt.colorbar(label='rgb')
plt.axis('off')

plt.tight_layout()
plt.show()
