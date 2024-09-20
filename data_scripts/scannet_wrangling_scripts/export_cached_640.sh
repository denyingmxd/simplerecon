#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

#python reader.py --scans_folder /data/laiyan/datasets/ScanNet/scans \
#--output_path  /data/laiyan/datasets/ScanNet/extracted/scans \
#--scan_list_file splits/scannetv2_train.txt \
#--num_workers 12 \
#--export_color_images \
#--rgb_resize 640 480

#python reader.py --scans_folder /data/laiyan/datasets/ScanNet/scans_test \
#--output_path  /data/laiyan/datasets/ScanNet/extracted/scans_test \
#--scan_list_file splits/scannetv2_test.txt \
#--num_workers 12 \
#--export_color_images \
#--rgb_resize 640 480

#python reader.py --scans_folder /data/laiyan/datasets/ScanNet/scans \
#--output_path  /data/laiyan/datasets/ScanNet/extracted/scans \
#--scan_list_file splits/scannetv2_val.txt \
#--num_workers 12 \
#--export_color_images \
#--rgb_resize 640 480