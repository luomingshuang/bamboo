import os
cwd = os.getcwd()                  ## get the current path

import sys
sys.path.append(cwd)               ## add local to package 

import os
import cv2
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from torchvision.ops.boxes import batched_nms

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from model import DeformableDETR
from train import get_parser
from nets.backbone import build_backbone
from nets.deformable_transformer import build_deformable_transformer

import local.misc as utils
from local.dataset_detr import build_dataset
from local.coco_eval import CocoEvaluator
from local import get_coco_api_from_dataset
from local import box_ops

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        
        boxes = boxes.cpu() * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


def main(args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    valid_dataset = build_dataset(image_set = "val", args=args)

    sampler_val = torch.utils.data.SequentialSampler(valid_dataset)

    batch_size = args.freeze_batch_size if args.freeze_train else args.unfreeze_batch_size

    valid_dl = DataLoader(
        valid_dataset, sampler=sampler_val,
        batch_size = batch_size, num_workers = args.num_workers,
        pin_memory=True, drop_last=False, collate_fn=utils.collate_fn)

    base_ds = get_coco_api_from_dataset(valid_dataset)
    postprocessors = {'bbox': PostProcess()}
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    backbone = build_backbone(args)
    transformer = build_deformable_transformer(args)

    num_classes = 91
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        device=device,
    )

    model.eval()
    
    checkpoint = torch.load(args.pretrained_model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], False)

    model.to(device)

    for batch in tqdm(valid_dl):
        images, targets = batch[0], batch[1]
        outputs = model(images.to(device))
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        coco_evaluator.update(res)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if "bbox" in postprocessors.keys():
        stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

    return stats, coco_evaluator    
 

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.test_results_dir:
        Path(args.test_results_dir).mkdir(parents=True, exist_ok=True)
    test_stats, coco_evaluator = main(args)

    log_stats = {**{f"test_{k}": v for k, v in test_stats.items()}}