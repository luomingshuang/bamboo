#!/usr/bin/env bash

set -x

export CUDA_VISIBLE_DEVICES='4'
world_size=1

COCO_PATH=/home/bcxiong1/data/coco2017/images
EXP_DIR=exps/r50_deformable_detr

python -u train.py \
    --output_dir ${EXP_DIR} \
    --coco_path ${COCO_PATH} \
    --eval_every 5
