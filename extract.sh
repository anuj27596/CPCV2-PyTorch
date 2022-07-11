#!/bin/bash

# python extract.py \
#     --dataset breastmnist --model_name_ext breastmnist \
#     --epochs 20 \
#     --crop 24-0 \
#     --encoder resnet18 \
#     --norm layer \
#     --grid_size 5 \
#     --pred_directions 4 \
#     --cpc_patch_aug \
#     --gray \
#     --model_num 10 \

python extract.py \
    --dataset retinamnist --model_name_ext retinamnist \
    --crop 24-0 \
    --encoder resnet18 \
    --norm layer \
    --grid_size 5 \
    --pred_directions 4 \
    --cpc_patch_aug \
    --model_num 100 \

# python extract.py \
#     --dataset bloodmnist --model_name_ext bloodmnist \
#     --crop 24-0 \
#     --encoder resnet18 \
#     --norm layer \
#     --grid_size 5 \
#     --pred_directions 4 \
#     --cpc_patch_aug \
#     --model_num 100 \
