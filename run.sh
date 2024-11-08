#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

exp_name="reproduce_dot_product_model_g4_b4_no_nan_jitter2"

python train.py --config_file configs/models/${exp_name}.yaml --data_config configs/data/scannet_default_train.yaml

python test.py \
--name ${exp_name} \
--output_base_path results \
--config_file configs/models/${exp_name}.yaml \
--load_weights_from_checkpoint logs/${exp_name}/version_0/checkpoints/last.ckpt \
--data_config configs/data/scannet_default_test.yaml \
--num_workers 8 \
--batch_size 4 \
--run_fusion \
--depth_fuser ours \
--dump_depth_visualization


exp_name="reproduce_small_dot_product_model_g4_b4_no_nan_jitter2"

python train.py --config_file configs/models/${exp_name}.yaml --data_config configs/data/scannet_default_train.yaml

python test.py \
--name ${exp_name} \
--output_base_path results \
--config_file configs/models/${exp_name}.yaml \
--load_weights_from_checkpoint logs/${exp_name}/version_0/checkpoints/last.ckpt \
--data_config configs/data/scannet_default_test.yaml \
--num_workers 8 \
--batch_size 4 \
--run_fusion \
--depth_fuser ours \
--dump_depth_visualization