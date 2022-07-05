#!/bin/bash

# python train_CPC.py \
#     --dataset breastmnist --model_name_ext breastmnist \
#     --epochs 10 \
#     --crop 24-0 \
#     --encoder resnet18 \
#     --norm layer \
#     --grid_size 5 \
#     --pred_steps 3 \
#     --pred_directions 4 \
#     --patch_aug \
#     --gray \
#     --neg_samples 8 \

python train_CPC.py \
    --dataset retinamnist --model_name_ext retinamnist \
    --epochs 10 \
    --crop 24-0 \
    --encoder resnet18 \
    --norm layer \
    --grid_size 5 \
    --pred_steps 3 \
    --pred_directions 4 \
    --patch_aug \
    --neg_samples 8 \
    --num_workers 2 \
    --trained_epochs 10 \
