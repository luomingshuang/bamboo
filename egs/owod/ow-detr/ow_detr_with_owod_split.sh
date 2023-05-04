#!/usr/bin/env bash

set -x

EXP_DIR=exps/OWOD_t1

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 2 \
    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWOD' --train_set 't1_train' --test_set 'all_task_test' --num_classes 81 \
    --unmatched_boxes --epochs 50 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'resnet50' --split owod

exit 1

EXP_DIR=exps/OWOD_t2

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 2 \
    --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWOD' --train_set 't2_train' --test_set 'all_task_test' --num_classes 81 \
    --unmatched_boxes --epochs 56 --top_unk 5 --lr 2e-5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t1/checkpoint0049.pth' \
    --split owod


EXP_DIR=exps/OWOD_t2_ft

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 2 \
    --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWOD' --train_set 't2_ft' --test_set 'all_task_test' --num_classes 81 \
    --unmatched_boxes --epochs 101 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t2/checkpoint0054.pth' \
    --split owod


EXP_DIR=exps/OWOD_t3

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 2 \
    --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWOD' --train_set 't3_train' --test_set 'all_task_test' --num_classes 81 \
    --unmatched_boxes --epochs 106 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t2_ft/checkpoint0099.pth' \
    --split owod

EXP_DIR=exps/OWOD_t3_ft

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 2 \
    --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWOD' --train_set 't3_ft' --test_set 'all_task_test' --num_classes 81 \
    --unmatched_boxes --epochs 136 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/dino_t3/checkpoint0104.pth' \
    --split owod


EXP_DIR=exps/OWOD_t4

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 2 \
    --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWOD' --train_set 't4_train' --test_set 'all_task_test' --num_classes 81 \
    --unmatched_boxes --epochs 141 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t3_ft/checkpoint0134.pth' \
    --split owod


EXP_DIR=exps/OWOD_t4_ft

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 2 \
    --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWOD' --train_set 't4_ft' --test_set 'all_task_test' --num_classes 81 \
    --unmatched_boxes --epochs 161 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t4/checkpoint0139.pth' \
    --split owod