#!/usr/bin/env bash

set -x

export CUDA_VISIBLE_DEVICES='4'
world_size=1

COCO_PATH=/home/bcxiong1/data/coco2017/images
EXP_DIR=exps/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage

python -u train.py \
    --output_dir ${EXP_DIR} \
    --coco_path ${COCO_PATH} \
    --eval_every 5 \
    --eval_model_path download/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth \
    --batch_size 1 \
    --num_queries 300 \
    --with_box_refine \
    --two_stage \
    --viz