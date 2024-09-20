#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

python reader.py --scans_folder /data/laiyan/datasets/ScanNet/scans \
--output_path  /data/laiyan/datasets/ScanNet/extracted/scans \
--scan_list_file splits/scannetv2_train.txt \
--num_workers 12 \
--export_poses \
--export_depth_images \
--export_color_images \
--export_intrinsics

python reader.py --scans_folder /data/laiyan/datasets/ScanNet/scans \
--output_path  /data/laiyan/datasets/ScanNet/extracted/scans \
--scan_list_file splits/scannetv2_val.txt \
--num_workers 12 \
--export_poses \
--export_depth_images \
--export_color_images \
--export_intrinsics