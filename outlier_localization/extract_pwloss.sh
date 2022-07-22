#!/bin/bash

python extract_patchwise_loss.py \
    --dataset retinamnist --model_name_ext retinamnist_healthy \
    --crop 24-0 \
    --encoder resnet18 \
    --norm layer \
    --grid_size 5 \
    --pred_steps 3 \
    --pred_directions 4 \
    --patch_aug \
    --batch_size 16 \
    --trained_epochs 200 \
