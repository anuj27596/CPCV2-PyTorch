#!/bin/bash

# python train_classifier.py \
#     --dataset breastmnist --model_name_ext breastmnist \
#     --epochs 10 \
#     --crop 24-0 \
#     --encoder resnet18 \
#     --norm layer \
#     --grid_size 5 \
#     --pred_directions 4 \
#     --cpc_patch_aug \
#     --gray \
#     --model_num 10 \
#     # --fully_supervised \

python train_classifier.py \
    --dataset retinamnist --model_name_ext retinamnist \
    --epochs 50 \
    --crop 24-0 \
    --encoder resnet18 \
    --norm layer \
    --grid_size 5 \
    --pred_directions 4 \
    --cpc_patch_aug \
    --model_num 100 \
    --fully_supervised \

# python train_classifier.py \
#     --dataset bloodmnist --model_name_ext bloodmnist \
#     --epochs 10 \
#     --crop 24-0 \
#     --encoder resnet18 \
#     --norm layer \
#     --grid_size 5 \
#     --pred_directions 4 \
#     --cpc_patch_aug \
#     --model_num 100 \
#     # --fully_supervised \
