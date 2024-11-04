import cv2
import numpy as np
from scipy.optimize import least_squares
import os
import matplotlib.pyplot as plt
from utils.generic_utils import read_image_file
import torch
import options
from tools import fusers_helper
from utils.dataset_utils import get_dataset
from utils.generic_utils import to_gpu, cache_model_outputs
from utils.metrics_utils import ResultsAverager, compute_depth_metrics_batched
from utils.visualization_utils import quick_viz_export, quick_viz_export_my
import torchvision.transforms.functional as TF
from pathlib import Path
def reverse_imagenet_normalize(image):
    """ Reverses ImageNet normalization in an input image. """

    image = TF.normalize(tensor=image,
        mean=(-2.11790393, -2.03571429, -1.80444444),
        std=(4.36681223, 4.46428571, 4.44444444))
    return image





def match_images(img1, img2):
    """
    Perform feature matching between two images
    Args:
        img1: First image (target)
        img2: Second image (source)
        detector_type: Type of feature detector ('SIFT' or 'ORB')
        ratio_thresh: Ratio test threshold for Lowe's ratio test
    Returns:
        pts1, pts2: Matched points in both images
        matches: List of matches
        (kp1, kp2): Keypoints in both images
    """
    # Convert images to uint8 if they are float
    if img1.dtype == np.float32 or img1.dtype == np.float64:
        img1 = (img1 * 255).astype(np.uint8)
    if img2.dtype == np.float32 or img2.dtype == np.float64:
        img2 = (img2 * 255).astype(np.uint8)

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    detector_type, ratio_thresh = 'SIFT', 0.7

    # Initialize detector
    if detector_type == 'SIFT':
        detector = cv2.SIFT_create(
            **{
                'nfeatures': 1000,  # No limit
                'nOctaveLayers': 4,  # More octave layers
                'contrastThreshold': 0.02,  # Lower threshold to detect more keypoints
                'edgeThreshold': 15,
                'sigma': 1.6
            }
        )  # Higher value = more edge features)

    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")

    # Find keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    if len(kp1) == 0 or len(kp2) == 0:
        return None, None, None, (None, None)

    # Match features using FLANN
    if detector_type == 'SIFT':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=52)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:  # ORB
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Perform matching
    matches = matcher.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    pts1 = []
    pts2 = []

    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    if len(good_matches) < 4:
        return None, None, None, (None, None)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    return pts1, pts2, good_matches, (kp1, kp2)


def show_matches(img1, img2, pts1, pts2, matches, kp1, kp2, title="Matches"):
    """
    Visualize matches between two images
    Args:
        img1, img2: Input images
        pts1, pts2: Matched points
        matches: List of matches
        kp1, kp2: Keypoints in both images
        title: Plot title
    """
    # Convert images to uint8 if they are float
    if img1.dtype == np.float32 or img1.dtype == np.float64:
        img1 = (img1 * 255).astype(np.uint8)
    if img2.dtype == np.float32 or img2.dtype == np.float64:
        img2 = (img2 * 255).astype(np.uint8)

    # Create figure
    plt.figure(figsize=(12, 6))

    # Draw matches
    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.imshow(match_img)
    plt.title(f"{title} - {len(matches)} matches")
    plt.axis('off')
    plt.show()
    return


def triangulate_to_depth_map(target_img_shape, pts1, pts2, target_pose, source_pose, K, epipolar_thresh):
    """
    Triangulate matched points to create a sparse depth map with epipolar error filtering.
    Also returns visualization data for epipolar geometry.
    """
    if pts1 is None or pts2 is None or len(pts1) == 0 or len(pts2) == 0 :

        return (
            np.zeros(target_img_shape),
            np.zeros(target_img_shape, dtype=bool),
            np.array([], dtype=bool),
            {}
        )

    # Calculate relative pose between cameras
    # source_pose and target_pose are world_T_cam
    relative_pose = np.linalg.inv(source_pose) @ target_pose  # source_cam_T_target_cam

    # Extract R and t from relative pose
    R = relative_pose[:3, :3]
    t = relative_pose[:3, 3]

    # Calculate essential matrix
    t_cross = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    E = t_cross @ R

    # Calculate fundamental matrix
    F = np.linalg.inv(K[:3, :3]).T @ E @ np.linalg.inv(K[:3, :3])

    # Reshape points for OpenCV function
    pts1_reshaped = pts1.reshape(-1, 1, 2)
    pts2_reshaped = pts2.reshape(-1, 1, 2)

    # Compute epipolar lines using OpenCV
    lines1 = cv2.computeCorrespondEpilines(pts2_reshaped, 2, F)
    lines2 = cv2.computeCorrespondEpilines(pts1_reshaped, 1, F)

    # Reshape lines
    lines1 = lines1.reshape(-1, 3)
    lines2 = lines2.reshape(-1, 3)

    # Calculate epipolar errors
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])

    # Distance from points to epipolar lines
    errors1 = np.abs(np.sum(pts1_h * lines1, axis=1)) / \
              np.sqrt(lines1[:, 0] ** 2 + lines1[:, 1] ** 2)
    errors2 = np.abs(np.sum(pts2_h * lines2, axis=1)) / \
              np.sqrt(lines2[:, 0] ** 2 + lines2[:, 1] ** 2)

    # Average symmetric epipolar distance
    epipolar_errors = (errors1 + errors2) / 2

    # Filter based on epipolar error
    epipolar_mask = epipolar_errors < epipolar_thresh

    # Get projection matrices
    P1 = K[:3, :3] @ np.hstack([np.eye(3), np.zeros((3, 1))])  # target camera
    P2 = K[:3, :3] @ np.hstack([R, t.reshape(3, 1)])  # source camera

    # Triangulate points
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts4d /= pts4d[3]
    pts3d = pts4d[:3].T

    # The points are already in target camera coordinate frame
    # Just extract Z coordinates as depths
    depths = pts3d[:, 2]

    # Combine all filters
    depth_mask = (depths > 0) & (depths < 5.)  # Adjust max depth as needed
    inlier_mask = depth_mask & epipolar_mask

    # Store visualization data
    viz_data = {
        'epipolar_lines1': lines1,
        'epipolar_lines2': lines2,
        'epipolar_errors': epipolar_errors,
        'depths': depths,
        'epipolar_mask': epipolar_mask,
        'depth_mask': depth_mask,
        'F': F
    }

    if not np.any(inlier_mask):
        return (
            np.zeros(target_img_shape),
            np.zeros(target_img_shape, dtype=bool),
            inlier_mask,
            viz_data
        )

    # Create sparse depth map
    sparse_depth = np.zeros(target_img_shape)
    valid_mask = np.zeros(target_img_shape, dtype=bool)

    # Convert points back to pixel coordinates for the depth map
    pts1_inliers = pts1[inlier_mask]
    depths_inliers = depths[inlier_mask]

    # Round pixel coordinates and ensure they're within image bounds
    pts1_px = np.round(pts1_inliers).astype(int)
    valid_pts = (pts1_px[:, 0] >= 0) & (pts1_px[:, 0] < target_img_shape[1]) & \
                (pts1_px[:, 1] >= 0) & (pts1_px[:, 1] < target_img_shape[0])

    # Fill in the sparse depth map
    pts1_px = pts1_px[valid_pts]
    depths_inliers = depths_inliers[valid_pts]

    sparse_depth[pts1_px[:, 1], pts1_px[:, 0]] = depths_inliers
    valid_mask[pts1_px[:, 1], pts1_px[:, 0]] = True

    return sparse_depth, valid_mask, inlier_mask, viz_data
def draw_epipolar_errors(target_img, source_img, pts1, pts2, viz_data, epipolar_thresh=1.0):
    """
    Draw epipolar lines and errors on the images.

    Args:
        target_img: Target image (H,W,3)
        source_img: Source image (H,W,3)
        pts1, pts2: Original point correspondences
        viz_data: Visualization data from triangulate_to_depth_map
        epipolar_thresh: Threshold used for epipolar error
    """
    import matplotlib.pyplot as plt

    # Create figure
    plt.figure(figsize=(15, 10))

    # Plot first image with epipolar lines
    plt.subplot(221)
    plt.imshow(target_img)
    plt.title('Target Image with Epipolar Lines')

    # Draw epipolar lines in first image
    for i in range(len(pts1)):
        color = 'g' if viz_data['epipolar_errors'][i] < epipolar_thresh else 'r'
        line = viz_data['epipolar_lines1'][i]
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [target_img.shape[1], -(line[2] + line[0] * target_img.shape[1]) / line[1]])
        plt.plot([x0, x1], [y0, y1], color, alpha=0.5)
        plt.plot(pts1[i, 0], pts1[i, 1], color + '.')

    # Plot second image with epipolar lines
    plt.subplot(222)
    plt.imshow(source_img)
    plt.title('Source Image with Epipolar Lines')

    # Draw epipolar lines in second image
    for i in range(len(pts2)):
        color = 'g' if viz_data['epipolar_errors'][i] < epipolar_thresh else 'r'
        line = viz_data['epipolar_lines2'][i]
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [source_img.shape[1], -(line[2] + line[0] * source_img.shape[1]) / line[1]])
        plt.plot([x0, x1], [y0, y1], color, alpha=0.5)
        plt.plot(pts2[i, 0], pts2[i, 1], color + '.')

    # Plot error histogram
    plt.subplot(223)
    plt.hist(viz_data['epipolar_errors'], bins=50)
    plt.axvline(epipolar_thresh, color='r', linestyle='--', label=f'Threshold ({epipolar_thresh}px)')
    plt.xlabel('Epipolar Error (pixels)')
    plt.ylabel('Count')
    plt.title('Distribution of Epipolar Errors')
    plt.legend()
    plt.grid(True)

    # Plot depth histogram
    plt.subplot(224)
    valid_depths = viz_data['depths'][viz_data['depth_mask']]
    plt.hist(valid_depths, bins=50)
    plt.xlabel('Depth (units)')
    plt.ylabel('Count')
    plt.title('Distribution of Valid Depths')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"Mean epipolar error: {viz_data['epipolar_errors'].mean():.3f} pixels")
    print(f"Median epipolar error: {np.median(viz_data['epipolar_errors']):.3f} pixels")
    print(f"Max epipolar error: {viz_data['epipolar_errors'].max():.3f} pixels")
    print(
        f"Epipolar inliers: {viz_data['epipolar_mask'].sum()}/{len(pts1)} ({100 * viz_data['epipolar_mask'].mean():.1f}%)")
    print(f"Depth inliers: {viz_data['depth_mask'].sum()}/{len(pts1)} ({100 * viz_data['depth_mask'].mean():.1f}%)")





def project_to_source(target_rgb, depth_map, K, target_world_T_cam, source_world_T_cam):
    """
    Project target RGB image to source view.

    Args:
        target_rgb: HxWx3 target RGB image
        depth_map: HxW depth map
        K: 4x4 or 3x3 intrinsic matrix
        target_world_T_cam: 4x4 world to camera transform for target view
        source_world_T_cam: 4x4 world to camera transform for source view

    Returns:
        warped_rgb: HxWx3 warped RGB image
        valid_mask: HxW boolean mask of valid projections
    """
    height, width = depth_map.shape
    K = K[:3, :3]  # Convert to 3x3 if 4x4

    # Generate pixel coordinates grid
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    pixels = np.stack([x, y, np.ones_like(x)], axis=-1)  # HxWx3

    # Back-project to 3D points in target camera space
    rays = np.linalg.inv(K) @ pixels.reshape(-1, 3).T  # 3xN
    points_target = rays * depth_map.reshape(-1)[None, :]  # 3xN

    # Convert to homogeneous coordinates
    points_target_h = np.vstack([points_target, np.ones((1, points_target.shape[1]))])  # 4xN

    # Transform points to world space then to source camera space
    target_cam_T_world = np.linalg.inv(target_world_T_cam)  # For target camera to world
    source_cam_T_world = np.linalg.inv(source_world_T_cam)  # For world to source camera

    # Combined transformation: source_cam_T_world @ world_T_target_cam @ target_points
    points_source = source_cam_T_world @ target_world_T_cam @ points_target_h  # 4xN

    # Project to source image
    points_source = points_source[:3] / points_source[2:3]  # 3xN
    pixels_source = (K @ points_source).T  # Nx3
    pixels_source = pixels_source[:, :2] / pixels_source[:, 2:]  # Nx2

    # Reshape back to image dimensions
    pixels_source = pixels_source.reshape(height, width, 2)

    # Check valid projections
    valid_mask = (points_source[2] > 0).reshape(height, width)  # Points in front of camera
    valid_mask &= (pixels_source[..., 0] >= 0) & (pixels_source[..., 0] < width - 1)  # X in bounds
    valid_mask &= (pixels_source[..., 1] >= 0) & (pixels_source[..., 1] < height - 1)  # Y in bounds

    # Warp image using grid_sample-like interpolation
    warped_rgb = np.zeros_like(target_rgb)

    # Convert pixels_source to integer indices and remainders for bilinear interpolation
    x = pixels_source[..., 0]
    y = pixels_source[..., 1]
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # Clip indices to image bounds
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)

    # Calculate interpolation weights
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # Interpolate
    warped_rgb[valid_mask] = (
            wa[valid_mask, None] * target_rgb[y0[valid_mask], x0[valid_mask]] +
            wb[valid_mask, None] * target_rgb[y1[valid_mask], x0[valid_mask]] +
            wc[valid_mask, None] * target_rgb[y0[valid_mask], x1[valid_mask]] +
            wd[valid_mask, None] * target_rgb[y1[valid_mask], x1[valid_mask]]
    )

    return warped_rgb, valid_mask


def show_warped_images(
                    target_img,
                    source_img,
                    depth_map,  # Using ground truth depth
                    intrinsics,
                    target_pose,
                    source_pose
                ):


    warped_target, valid_mask = project_to_source(
        target_rgb=target_img,
        depth_map=depth_map,  # Using ground truth depth
        K=intrinsics,
        target_world_T_cam=target_pose,
        source_world_T_cam=source_pose
    )

    # Visualize results
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(target_img)
    plt.title('Target Image')
    plt.subplot(132)
    plt.imshow(warped_target)
    plt.title('Warped Target')
    plt.subplot(133)
    plt.imshow(source_img)
    plt.title('Source Image')
    plt.show()
    return


from tqdm import tqdm
def main(opts):
    # get dataset
    dataset_class, scans = get_dataset(opts.dataset, opts.dataset_scan_split_file, opts.single_debug_scan_id)

    # path where results for this model, dataset, and tuple type are.
    results_path = os.path.join(opts.output_base_path, opts.name, opts.dataset, opts.frame_tuple_type)


    with torch.inference_mode():

        # loop over scans
        for scan in (scans):
            scan = 'scene0707_00'
            # initialize fuser if we need to fuse

            # set up dataset with current scan
            dataset = dataset_class(
                opts.dataset_path,
                split=opts.split,
                mv_tuple_file_suffix=opts.mv_tuple_file_suffix,
                limit_to_scan_id=scan,
                include_full_res_depth=True,
                tuple_info_file_location=opts.tuple_info_file_location,
                num_images_in_tuple=None,
                shuffle_tuple=opts.shuffle_tuple,
                include_high_res_color=(
                        (opts.fuse_color and opts.run_fusion)
                        or opts.dump_depth_visualization
                ),
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

            import time
            times= []
            ratios_meds = []
            num_features = []
            for batch_ind, batch in enumerate(tqdm(dataloader)):
                # get data, move to GPU
                # if batch_ind!=111:
                #     continue
                cur_data, src_data = batch
                # cur_data = to_gpu(cur_data, key_ignores=["frame_id_string"])
                # src_data = to_gpu(src_data, key_ignores=["frame_id_string"])
                depth_gt = cur_data["depth_b1hw"]
                #cam_T_world_b44, world_T_cam_b44
                target_img =reverse_imagenet_normalize(cur_data["image_b3hw"])[0].cpu().numpy().transpose(1, 2, 0)
                source_imgs = [reverse_imagenet_normalize(src_data["image_b3hw"])[0][i].cpu().numpy().transpose(1, 2, 0) for i in range(len(src_data["image_b3hw"][0]))]
                target_img = cv2.resize(target_img, (depth_gt.shape[-1], depth_gt.shape[-2]), interpolation=cv2.INTER_LINEAR)
                source_imgs = [cv2.resize(source_img, (depth_gt.shape[-1], depth_gt.shape[-2]), interpolation=cv2.INTER_LINEAR) for source_img in source_imgs]
                target_pose = cur_data["world_T_cam_b44"][0].cpu().numpy()
                source_poses = [src_data["world_T_cam_b44"][0][i].cpu().numpy() for i in range(len(src_data["world_T_cam_b44"][0]))]
                intrinsics = cur_data["K_s0_b44"][0].cpu().numpy()
                depth_gt = depth_gt[0][0].cpu().numpy()

                 # this block serves as match and show matches
                # Get matches between target and first source image
                start = time.time()
                source_img = source_imgs[0]
                source_pose = source_poses[0]
                # Basic feature matching
                pts1, pts2, matches, (kp1, kp2) = match_images(
                    target_img,
                    source_img,
                )

                # show_matches(target_img, source_img, pts1, pts2, matches, kp1, kp2, "Basic Matches")

                # show_warped_images(
                #     target_img=target_img,
                #     source_img=source_img,
                #     depth_map=depth_gt,  # Using ground truth depth
                #     intrinsics=intrinsics,
                #     target_pose=target_pose,
                #     source_pose=source_pose
                # )


                # print(123)

                sparse_depth, valid_mask, inlier_mask, viz_data = triangulate_to_depth_map(
                    target_img_shape=depth_gt.shape,
                    pts1=pts1,
                    pts2=pts2,
                    target_pose=target_pose,
                    source_pose=source_pose,
                    K=intrinsics,
                    epipolar_thresh=2.0
                )
                num_features.append(inlier_mask.sum())
                end = time.time()
                times.append(end-start)
                # print(times)
                # draw_epipolar_errors(
                #     target_img=target_img,
                #     source_img=source_img,
                #     pts1=pts1,
                #     pts2=pts2,
                #     viz_data=viz_data,
                #     epipolar_thresh=2.0
                # )
                final_valid_mask = valid_mask  & (sparse_depth > 0.5) & (~ np.isnan(depth_gt)) & (sparse_depth < 5.0)
                print(batch_ind, cur_data['frame_id_string'][0], final_valid_mask.sum(),valid_mask.sum())
                sparse_depth_valid = sparse_depth * final_valid_mask
                gt_depth_valid = depth_gt * final_valid_mask
                ratios = gt_depth_valid[final_valid_mask] / sparse_depth_valid[final_valid_mask]
                ratio_med = np.median(ratios)
                ratios_meds.append(ratio_med)
                file_path = os.path.join('/data/laiyan/datasets/ScanNet/sparse_depth','scans_test',scan,'sensor_data','frame-{}.sparse_depth.256.npz'.format(cur_data['frame_id_string'][0]))
                save_path = Path(file_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(save_path, sparse_depth_valid)
                # exit()
            plt.hist(ratios_meds,bins=50)
            plt.show()
            print(np.mean(ratios_meds),np.std(ratios_meds))

            exit()
#


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

