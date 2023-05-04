#!/usr/bin/env bash

set -x

export CUDA_VISIBLE_DEVICES='2,3'
world_size=2

EXP_DIR=exps/OWDETR_t1

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR' --train_set 't1_train' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 45 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --split 'owdetr' \
    --world_size ${world_size}

EXP_DIR=exps/OWDETR_t2

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
    --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR' --train_set 't2_train' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 50 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWDETR_t1/checkpoint0044.pth' \
    --split 'owdetr' \
    --world_size ${world_size}

EXP_DIR=exps/OWDETR_t2_ft

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
    --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR' --train_set 't2_ft' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 100 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWDETR_t2/checkpoint0049.pth' \
    --split 'owdetr' \
    --world_size ${world_size}

EXP_DIR=exps/OWDETR_t3

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
    --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR' --train_set 't3_train' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 106 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWDETR_t2_ft/checkpoint0099.pth' \
    --split 'owdetr' \
    --world_size ${world_size}

EXP_DIR=exps/OWDETR_t3_ft

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
    --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR' --train_set 't3_ft' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 161 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWDETR_t3/checkpoint0104.pth' \
    --split 'owdetr' \
    --world_size ${world_size}

EXP_DIR=exps/OWDETR_t4

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
    --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR' --train_set 't4_train' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 171 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWDETR_t3_ft/checkpoint0159.pth' \
    --split 'owdetr' \
    --world_size ${world_size}

EXP_DIR=exps/OWDETR_t4_ft

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
    --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR' --train_set 't4_ft' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 302 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWDETR_t4/checkpoint0169.pth' \
    --split 'owdetr' \
    --world_size ${world_size}