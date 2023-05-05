# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import os
import logging
import argparse
import datetime
import json
import random
import time
import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from typing import Optional, Tuple

from icefall.checkpoint import load_checkpoint
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    setup_logger,
    str2bool,
)

import datasets
import util.misc as utils
import datasets.samplers as samplers
from util.metric_tracker import MetricsTracker
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.data_prefetcher import data_prefetcher, to_cuda
from engine import evaluate, train_one_epoch, viz
from models import build_model

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument("--tensorboard", type=str2bool, default=True, help="use tensorboard or not.")

    # epochs and batch size, optimize methods
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")  
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    
    # dataset parameters
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--dataset_file', type=str, default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_model_path', default='', help='the model for evaling')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--eval_every', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pretrain', default='', help='initialized from the pre-training model')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # distributed training parameters
    parser.add_argument('--distributed', type=bool, default=True, help='use distributed method for training or not')
    parser.add_argument('--world_size', type=int, default=1, help='the number of gpus for ddp training')
    parser.add_argument('--master_port', type=int, default=12345, help='master port to use for ddp training')

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    is saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 10,
            "reset_interval": 200,
            "valid_interval": 10, #3000,
            "save_checkpoint_interval": 5,
            "env_info": get_env_info(),
        }
    )

    return params


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> None:
    """Load checkpoint from file.

    If params.start_epoch is positive, it will load the checkpoint from
    `params.start_epoch - 1`. Otherwise, this function does nothing.

    Apart from loading state dict for `model`, `optimizer` and `scheduler`,
    it also updates `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The learning rate scheduler we are using.
    Returns:
      Return None.
    """
    if params.start_epoch <= 0 and not params.pretrain:
        return

    filename = Path(params.output_dir) / f"epoch-{params.start_epoch-1}.pt"
    
    if params.pretrain:
        filename = params.pretrain
    
    saved_params = load_checkpoint(
        filename,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
    """
    if rank != 0:
        return
    filename = Path(params.output_dir) / f"epoch-{params.cur_epoch}.pt"

    if params.cur_epoch % 5 == 0:
        save_checkpoint_impl(
            filename=filename,
            model=model,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=rank,
        )
    
    if isinstance(model, DDP):
        model = model.module
    
    checkpoint = {"model": model.state_dict()}

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = Path(params.output_dir) / "best-train-loss.pt"
        torch.save(checkpoint, best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = Path(params.output_dir) / "best-valid-loss.pt"
        torch.save(checkpoint, best_valid_filename)


def compute_loss(
    params: AttributeDict,
    model: nn.Module,
    criterion: nn.Module,
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. 
      criterion:
        The criterion for training. 
      batch:
        A batch of data. 
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
    """
    samples, targets = batch[0], batch[1]
    
    loss_dict = {}
    losses = 0

    info = MetricsTracker()
    
    with torch.set_grad_enabled(is_training):
        # Losses: loss_ce, class_error, loss_bbox, loss_giou, cardinality_error
        # loss_ce: classification loss NLL
        # class_error: compute the precision@k for the specified values of k, just for log
        # loss_bbox: the L1 regression loss
        # loss_giou: the giou loss for object detection
        # cardinality_error: compute the cardinality error, the absolute error in the number of 
        # predicted non-empty boxes, not a real loss, just for log 
        # Note: here, we don't talk about segmentation task, so there is no loss_masks

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = deepcopy(criterion.weight_dict)
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        ## Just printing NOt affecting in loss function
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
 
        loss_value = losses_reduced_scaled.item()

        assert losses.requires_grad == is_training


        info["loss_reduced_scaled_value"] = loss_value
    
        # just reduce, not scale
        for k in loss_dict_reduced_unscaled.keys():
            info[k] = loss_dict_reduced_unscaled[k].detach().cpu().item()

        # reduce and scale
        for k in loss_dict_reduced_scaled.keys():
            info[k] = loss_dict_reduced_scaled[k].detach().cpu().item()
    
    return losses, info


def compute_validation_loss(
    params: AttributeDict,
    model: nn.Module,
    criterion: nn.Module,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process. The validation loss
    is saved in `params.valid_loss`.
    """
    model.eval()

    tot_loss = MetricsTracker()

    device = params.device

    for batch_idx, batch in enumerate(tqdm(valid_dl)):
        if batch_idx >=20: break
        samples = batch[0].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch[1]]
        batch = [samples, targets]
        losses, loss_info = compute_loss(
            params=params,
            model=model,
            criterion=criterion,
            batch=batch,
            is_training=False,
        )

        assert losses.requires_grad is False

        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(losses.device)

    loss_value = tot_loss["loss_reduced_scaled_value"]

    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      criterion:
        The criterion for deformable-detr.
      optimizer:
        The optimizer we are using.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
        
    """
    model.train()
    criterion.train()

    tot_loss = MetricsTracker()
    device = params.device

    prefetcher = data_prefetcher(train_dl, device, prefetch=True)
    # samples, targets = batch
    batch = prefetcher.next()

    for batch_idx in range(len(train_dl)):
        if batch_idx >=20: break
        params.batch_idx_train += 1
        batch_size = batch[0].tensors.size(0)

        losses, loss_info = compute_loss(
            params=params,
            model=model,
            criterion=criterion,
            batch=batch,
            is_training=True,
        )
    
        # summary stats.
        tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

        optimizer.zero_grad()
        losses.backward()
        if params.clip_max_norm > 0:
            clip_grad_norm_(model.parameters(), params.clip_max_norm)
        optimizer.step()
      
        if batch_idx % params.log_interval == 0:
            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}"
            )
        if batch_idx % params.log_interval == 0:

            if tb_writer is not None:
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)

        if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                criterion=criterion,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation {valid_info}")
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer,
                    "train/valid_",
                    params.batch_idx_train,
                )
        
        batch = prefetcher.next()

    loss_value = tot_loss["loss_reduced_scaled_value"]
    params.train_loss = loss_value

    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    
    if world_size > 1:
        setup_dist(rank, world_size, args.master_port)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    params.device = device

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{Path(params.output_dir)}/tensorboard")
    else:
        tb_writer = None

    setup_logger(f"{Path(args.output_dir)}/log/log-train")
    logging.info("Training started")    
    logging.info(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    checkpoints = load_checkpoint_if_available(params=params, model=model)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    
    if args.distributed and world_size > 1:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed and world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    elif args.dataset_file == "coco":
        base_ds = get_coco_api_from_dataset(dataset_val)
    else:
        base_ds = dataset_val

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    if params.start_epoch > 1:
        if checkpoints and "optimizer" in checkpoints:
            logging.info("Loading optimizer state dict")
            optimizer.load_state_dict(checkpoints["optimizer"])
        if (
            checkpoints
            and "scheduler" in checkpoints
            and checkpoints["scheduler"] is not None
        ):
            logging.info("Loading scheduler state dict")
            scheduler.load_state_dict(checkpoints["scheduler"])

    if args.eval_model_path:
        checkpoint = torch.load(args.eval_model_path)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)     
        if args.output_dir:
            if args.eval:
                test_stats, coco_evaluator = evaluate(model_without_ddp, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir)
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pt")
            if args.viz:
                viz(model_without_ddp, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir)
        return        

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if epoch > params.start_epoch:
            logging.info(f"epoch {epoch}, lr: {scheduler.get_last_lr()[0]}")

        if tb_writer is not None:
            tb_writer.add_scalar(
                "train/lr",
                scheduler.get_last_lr()[0],
                params.batch_idx_train,
            )
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_stats = train_one_epoch(
            params=params,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_dl=data_loader_train,
            valid_dl=data_loader_val,
            tb_writer=tb_writer,
            world_size=world_size,
        )

        scheduler.step()

        save_checkpoint(
            params=params,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=rank,
        )

        if args.output_dir and epoch % args.eval_every == 0:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )
            log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log_test.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pt']
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    logging.info('Training time {}'.format(total_time_str))

    logging.info('Done!')
    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = argparse.ArgumentParser('Deformable-DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)

if __name__ == '__main__':
    main()