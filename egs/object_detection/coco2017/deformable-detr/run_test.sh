# Parameters as default
export CUDA_VISIBLE_DEVICES="0" 
python deformable-detr/test.py \
    --pretrained-model-path deformable-detr/download/r50_deformable_detr-checkpoint.pth
    --test-samples-dir samples \
    --test-results-dir deformable-detr/exp/test_samples_results


# with box refine and two stages
export CUDA_VISIBLE_DEVICES="0" 
python deformable-detr/test.py \
    --pretrained-model-path deformable-detr/download/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth \
    --with-box-refine \
    --two-stage \
    --test-samples-dir samples \
    --test-results-dir deformable-detr/exp/test_samples_results

