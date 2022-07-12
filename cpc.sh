#!/bin/bash

# python train_CPC.py \
#     --dataset breastmnist --model_name_ext breastmnist \
#     --epochs 100 \
#     --crop 24-0 \
#     --encoder resnet18 \
#     --norm layer \
#     --grid_size 5 \
#     --pred_steps 3 \
#     --pred_directions 4 \
#     --patch_aug \
#     --gray \
#     --neg_samples 8 \

# python train_CPC.py \
#     --dataset retinamnist --model_name_ext retinamnist_temp \
#     --epochs 100 \
#     --crop 24-0 \
#     --encoder resnet18 \
#     --norm layer \
#     --grid_size 5 \
#     --pred_steps 3 \
#     --pred_directions 4 \
#     --patch_aug \
#     --neg_samples 8 \
#     --num_workers 2 \
#     # --trained_epochs 10 \

# python train_CPC.py \
#     --dataset bloodmnist --model_name_ext bloodmnist \
#     --epochs 100 \
#     --crop 24-0 \
#     --encoder resnet18 \
#     --norm layer \
#     --grid_size 5 \
#     --pred_steps 3 \
#     --pred_directions 4 \
#     --patch_aug \
#     --neg_samples 8 \
#     --num_workers 2 \
#     # --trained_epochs 10 \

python train_CPC.py \
    --dataset kdr --model_name_ext kdr \
    --epochs 1 \
    --crop 224-16 \
    --encoder resnet18 \
    --norm layer \
    --grid_size 7 \
    --pred_steps 5 \
    --pred_directions 4 \
    --patch_aug \
    --neg_samples 8 \
    --num_workers 2 \
    --batch_size 1 \
    # --trained_epochs 10 \
