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
import time
from tools.mesh_renderer import (DEFAULT_CAM_FRUSTUM_MATERIAL,
                                 DEFAULT_MESH_MATERIAL, Renderer,
                                 SmoothBirdsEyeCamera, camera_marker,
                                 create_light_array, get_image_box,
                                 transform_trimesh)

import options
import torch
from utils.generic_utils import to_gpu
from utils.dataset_utils import get_dataset
from tqdm import tqdm
from utils.generic_utils import reverse_imagenet_normalize

def main(opts):
    opts.batch_size = 1

    # get dataset
    dataset_class, scans = get_dataset(opts.dataset,
                        opts.dataset_scan_split_file, opts.single_debug_scan_id)

    opts.split = 'train'
    opts.jitter_type = 1

    with torch.inference_mode():
        for scan in (scans):
            dataset = dataset_class(
                opts.dataset_path,
                split=opts.split,
                mv_tuple_file_suffix=opts.mv_tuple_file_suffix,
                limit_to_scan_id=scan,
                include_full_res_depth=True,
                tuple_info_file_location=opts.tuple_info_file_location,
                num_images_in_tuple=None,
                shuffle_tuple=opts.shuffle_tuple,
                include_high_res_color=opts.fuse_color and opts.run_fusion,
                include_full_depth_K=True,
                skip_frames=opts.skip_frames,
                skip_to_frame=opts.skip_to_frame,
                image_width=opts.image_width,
                image_height=opts.image_height,
                pass_frame_id=True,
                opts=opts
            )

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=opts.batch_size,
                shuffle=False,
                num_workers=opts.num_workers,
                drop_last=False,
            )

            render_depth_height, render_depth_width = dataset.depth_height, dataset.depth_width
            render_rgb_height, render_rgb_width = dataset.image_height, dataset.image_width

            fpv_depth_renderer = Renderer(height=render_depth_height, width=render_depth_width)
            fpv_rgb_renderer = Renderer(height=render_rgb_height, width=render_rgb_width)

            mesh_gt_path = dataset.get_gt_mesh_path(opts.dataset_path, opts.split, scan)
            mesh_gt = trimesh.load(mesh_gt_path, force='mesh')
            meshes = [mesh_gt]

            for batch_ind, batch in enumerate((dataloader)):
                print(batch_ind)
                # get data, move to GPU
                cur_data, src_data = batch
                if "frame_id_string" in cur_data:
                    frame_id = cur_data["frame_id_string"][0]
                else:
                    frame_id = f"{str(batch_ind):6d}"

                # get the mesh

                intrinsics = cur_data["K_s0_b44"][0]

                world_T_cam = cur_data["world_T_cam_b44"][0]
                depth_gt = cur_data["depth_b1hw"][0][0]
                rgb = reverse_imagenet_normalize(cur_data["image_b3hw"][0]).permute(1, 2, 0)

                rgb = (rgb * 255).numpy().astype(np.uint8)

                rendered_depth = fpv_depth_renderer.render_mesh(
                                                meshes,
                                                render_depth_height, render_depth_width,
                                                world_T_cam, intrinsics,
                                                False,
                                            )

                intrinsics_rgb = intrinsics.copy()
                intrinsics_rgb[0] *= render_rgb_width / render_depth_width
                intrinsics_rgb[1] *= render_rgb_height / render_depth_height
                rendered_color = fpv_rgb_renderer.render_mesh(
                                                meshes,
                                                render_rgb_height, render_rgb_width,
                                                world_T_cam, intrinsics_rgb,
                                                True,
                                            )
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(rendered_depth, cmap='jet',vmin=0.,vmax=3)
                plt.title('Rendered Depth')
                plt.colorbar(label='Depth')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(depth_gt, cmap='jet',vmin=0.,vmax=3)
                plt.title('Original Depth')
                plt.colorbar(label='Depth')
                plt.axis('off')

                diff = np.abs(rendered_depth - depth_gt)
                plt.subplot(1, 3, 3)
                plt.imshow(diff, cmap='jet',vmin=0,vmax=0.2)
                plt.title('diff')
                plt.colorbar(label='diff')
                plt.axis('off')

                plt.tight_layout()
                plt.show()

                plt.figure(figsize=(15, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(rendered_color)
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



    return 0


if __name__ == '__main__':
    # don't need grad for test.
    torch.set_grad_enabled(False)

    # get an instance of options and load it with config file(s) and cli args.
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    option_handler.pretty_print_options()
    print("\n")
    opts = option_handler.options

    # if no GPUs are available for us then, use the 32 bit on CPU
    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32
    main(opts)