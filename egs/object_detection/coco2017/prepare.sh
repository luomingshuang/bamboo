#!/usr/bin/env bash

set -eou pipefail

stage=-1
stop_stage=100

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/voc2007_model_data
#      You can find the files for backbone weights, style of 
#      text, voc 2007 classes
#      You can also download these files form this url:
#      https://zenodo.org/record/3625687#.Ybn7HagzY2w. 
#     
#     - voc_weights_backbone
#       - resnet50-19c8e357.pth
#       - vgg16-397923af.pth
#       - voc_weights_resnet.pth
#       - voc_weights_vgg.pth
#     - semhei.ttf
#     - voc_classes.txt
#       

dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  log "Stage -1: Download VOC2007 dataset and Unzip"
  # We assume that you have downloaded the VOC2007 dataset, and unzip it.
  # wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  # wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  # wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
  # tar xvf VOCtrainval_06-Nov-2007.tar
  # tar xvf VOCtest_06-Nov-2007.tar
  # tar xvf VOCdevkit_08-Jun-2007.tar
  
  # Here, we get the VOC2007 dataset directory as follows:
  # - /userhome/data/voc_2007_2012/VOC2007
  #   - Annotations  
  #   - ImageSets  
  #   - JPEGImages  
  #   - SegmentationClass  
  #   - SegmentationObject 
  #   - codes  
  #   - test
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download model data"
  # For this step, you should download some model data files for modeling.
  # You can download them as follows.
  #
  
  [ ! -e $dl_dir/voc2007_model_data ] && mkdir -p $dl_dir/voc2007_model_data

  # Download the model data
  # You should make sure you have installed git-lfs
  # You can install git-lfs as follows:
  # apt-get install git-lfs
  if [! -e $dl_dir/voc2007_model_data ]; then
      git lfs install
      git clone https://huggingface.co/luomingshuang/voc2007_model_data $dl_dir/voc2007_model_data
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare train and val data"
  data_dir=/userhome/data/voc_2007_2012
  out_dir=data/voc2007
  year=2007
  
  mkdir -p $out_dir

  python ./local/voc_annotation.py \
    --data-dir $data_dir \
    --out-dir $out_dir \
    --class-txt $dl_dir/voc2007_model_data/voc_classes.txt \
    --year 2007 \
    --trainval-percent 0.9 \
    --train-percent 0.9 \
    --annotation-mode 0
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Train the faster rcnn"
  # Just a simple training commands
  # You should change your parameters by yourself.
  python ./faster_rcnn/train.py \
    --exp-dir faster_rcnn/exp \
    --wandb True \
    --freeze-train True \
    --freeze-batch-size 16 \
    --unfreeze-batch-size 12 \
    --num-epochs 100 \
    --freeze-epoch 50
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Eval metrics for the trained model"
  # Eval metrics for the voc2007 test data
  
  python ./faster_rcnn/eval_metrics.py \
    --model-path faster_rcnn/exp/best-valid-loss.pt \
    --classes-path $dl_dir/voc2007_model_data/voc_classes.txt \
    --backbone resnet50 \
    --confidence 0.5 \
    --nms-iou 0.3 \
    --anchors-size [8, 16, 32] \
    --map-mode 0 \ 
    --voc-data-path /userhome/data/voc_2007_2012
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Do summary for the trained model"
  
  python ./faster_rcnn/summary.py \
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Test some samples on the trained model"
  
  python ./faster_rcnn/test.py \
    --model-path faster_rcnn/exp/best-valid-loss.pt \
    --classes-path $dl_dir/voc2007_model_data/voc_classes.txt \
    --backbone resnet50 \
    --confidence 0.5 \
    --nms-iou 0.3 \
    --anchors-size [8, 16, 32] \
    --mode dir_predict \
    --samples-dir-path samples\ \
    --detect-results-path faster_rcnn/exp/detect_samples_results
fi