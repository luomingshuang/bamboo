#!/usr/bin/env bash

set -x

export CUDA_VISIBLE_DEVICES='0,1,2,3'
world_size=4

EXP_DIR_PRE=exps_test

EXP_DIR=${EXP_DIR_PRE}/OWDETR_t1

python -u train.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR' --train_set 't1_train' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 46 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --split 'owdetr' \
    --world_size ${world_size} \
    --num_workers 0

EXP_DIR=${EXP_DIR_PRE}/OWDETR_t2

python -u train.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR' --train_set 't2_train' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 51 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain ${EXP_DIR_PRE}/OWDETR_t1/epoch-45.pt \
    --split 'owdetr' \
    --world_size ${world_size} \
    --num_workers 0

EXP_DIR=${EXP_DIR_PRE}/OWDETR_t2_ft

python -u train.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR' --train_set 't2_ft' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 101 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain ${EXP_DIR_PRE}/OWDETR_t2/epoch-50.pt \
    --split 'owdetr' \
    --world_size ${world_size} \
    --num_workers 0

EXP_DIR=${EXP_DIR_PRE}/OWDETR_t3

python -u train.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR' --train_set 't3_train' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 106 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain ${EXP_DIR_PRE}/OWDETR_t2_ft/epoch-100.pt \
    --split 'owdetr' \
    --world_size ${world_size} \
    --num_workers 0

EXP_DIR=${EXP_DIR_PRE}/OWDETR_t3_ft

python -u train.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR' --train_set 't3_ft' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 161 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain ${EXP_DIR_PRE}/OWDETR_t3/epoch-105.pt \
    --split 'owdetr' \
    --world_size ${world_size} \
    --num_workers 0

EXP_DIR=${EXP_DIR_PRE}/OWDETR_t4

python -u train.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR' --train_set 't4_train' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 171 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain ${EXP_DIR_PRE}/OWDETR_t3_ft/epoch-160.pt \
    --split 'owdetr' \
    --world_size ${world_size} \
    --num_workers 0

EXP_DIR=${EXP_DIR_PRE}/OWDETR_t4_ft

python -u train.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR' --train_set 't4_ft' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 305 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain ${EXP_DIR_PRE}/OWDETR_t4/epoch-170.pt \
    --split 'owdetr' \
    --world_size ${world_size} \
    --num_workers 0