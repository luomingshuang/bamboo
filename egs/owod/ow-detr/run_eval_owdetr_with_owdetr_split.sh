set -x

export CUDA_VISIBLE_DEVICES='0'
world_size=1

EXP_DIR_PRE=exps

EXP_DIR=${EXP_DIR_PRE}/OWDETR_t1

python -u train.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR' --train_set 't1_train' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 45 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --eval_model_path ${EXP_DIR_PRE}/OWDETR_t1/best-train-loss.pt --eval \
    --split 'owdetr' \
    --world_size ${world_size}


EXP_DIR=${EXP_DIR_PRE}/OWDETR_t2_ft

python -u train.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 --data_root '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR' --train_set 't2_ft' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 101 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --eval_model_path ${EXP_DIR_PRE}/OWDETR_t2_ft/best-train-loss.pt --eval \
    --split 'owdetr' \
    --world_size ${world_size}