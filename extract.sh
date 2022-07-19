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

# python extract.py \
#     --dataset retinamnist --model_name_ext retinamnist \
#     --crop 24-0 \
#     --encoder resnet18 \
#     --norm layer \
#     --grid_size 5 \
#     --pred_directions 4 \
#     --cpc_patch_aug \
#     --model_num 100 \

# python extract.py \
#     --dataset retinamnist --model_name_ext retinamnist_pathological \
#     --crop 24-0 \
#     --encoder resnet18 \
#     --norm layer \
#     --grid_size 5 \
#     --pred_directions 4 \
#     --cpc_patch_aug \
#     --model_num 100 \

# python extract.py \
#     --dataset bloodmnist --model_name_ext bloodmnist_gray \
#     --crop 24-0 \
#     --encoder resnet18 \
#     --norm layer \
#     --grid_size 5 \
#     --pred_directions 4 \
#     --cpc_patch_aug \
#     --model_num 100 \
#     --gray \

python extract.py \
    --dataset kdr --model_name_ext kdr \
    --crop 224-16 \
    --encoder resnet18 \
    --norm layer \
    --grid_size 7 \
    --pred_directions 4 \
    --cpc_patch_aug \
    --model_num 5 \
    --batch_size 8 \
    --num_workers 4 \
