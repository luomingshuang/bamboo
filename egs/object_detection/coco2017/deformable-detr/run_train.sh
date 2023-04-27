cd ..


# not with box refine, not with two stages, not with dilation convolution
# with num-feature-levels = 4
export CUDA_VISIBLE_DEVICES='0,1,2,3'
python deformable-detr/train.py \
    --world-size 4


# or if just with single scale feature map
export CUDA_VISIBLE_DEVICES='0,1,2,3'
python deformable-detr/train.py \
    --world-size 4 \
    --num-feature-levels 1


# or if with single scale feature map and dilation convolution
export CUDA_VISIBLE_DEVICES='0,1,2,3'
python deformable-detr/train.py \
    --world-size 4 \
    --num-feature-levels 1 \
    --dilation 


# or if with box refine
export CUDA_VISIBLE_DEVICES='0,1,2,3'
python deformable-detr/train.py \
    --with-box-refine


# or if with box refine and two stages:
export CUDA_VISIBLE_DEVICES='0,1,2,3'
python deformable-detr/train.py \
    --with-box-fefine \
    --two-stage

