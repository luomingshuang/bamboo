export CUDA_VISIBLE_DEVICES='0,1,2,3'
## with aux loss to accelerate training
python detr/train.py \
    --coco-path ~/coco2017 \
    --unfreeze-batch-size 16 \
    --freeze-train False \
    --separate-optimize True


export CUDA_VISIBLE_DEVICES='0'
python detr/test.py \
    --test-samples-dir samples \
    --test-results-dir detr/exp/test_samples_results \
    --pretrained-model-path detr/exp/best-valid-loss.pt


export CUDA_VISIBLE_DEVICES='0'
python detr/generate_predict_json.py

python detr/eval_metrics_mscoco_api.py

or

python detr/eval_metrics.py
