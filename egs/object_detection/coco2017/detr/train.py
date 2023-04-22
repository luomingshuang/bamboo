#!/usr/bin/env python3
# Copyright      2023        (authors:   Mingshuang Luo)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Usage:
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
  python detr/train.py \
     --world-size 4 \
     --full-libri 1 \
     --max-duration 300 \
     --num-epochs 20
"""

import os
cwd = os.getcwd()                  ## get the current path

import sys
sys.path.append(cwd)               ## add local to package 

from icefall.checkpoint import load_checkpoint
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    setup_logger,
    str2bool,
)

import local.misc as utils
from local.utils import MetricsTracker
from local.utils import get_classes_mscoco2017
from local.dataset_detr import build_dataset

from model import build_model

import argparse
import logging
import numpy as np
from pathlib import Path
from shutil import copyfile
from typing import Optional, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler

from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset-file",
        type=str,
        default="coco",
        help="the dataset name for using"
    )
    parser.add_argument(
        "--coco-path",
        type=str,
        default="~/data/coco2017/images",
        help="Path for the coco2017 dataset directory."
    )

    parser.add_argument(
        "--mscoco2017-classes-path",
        type=str,
        default="download/model_data/mscoco2017_classes.txt",
        help="Path for the mscoco2017_classes.txt file.",
    )

    parser.add_argument(
        "--test-samples-dir",
        type=str,
        default="samples",
        help="The directory path for testing samples."
    )

    parser.add_argument(
        "--test-results-dir",
        type=str,
        default="detr/exp/test_samples_results",
        help="The directory path for testing samples'results."
    )

    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default="download/model_data/pretrained_model_weights/detr-r50-e632da11-mscoco2017.pth",
        help="The path for the pretrained model path."
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="detr/exp",
        help="Path for the pretrained model weights file, eg: epoch-0.pt.",
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--freeze-train",
        type=bool,
        default=True,
        help="train the model by freezing the model firstly or not.",
    )

    parser.add_argument(
        "--freeze-epoch",
        type=int,
        default=50,
        help="the training epochs for freezing the backbone weights.",
    )

    parser.add_argument(
        "--freeze-batch-size",
        type=int,
        default=4,
        help="the batch size for training when freeze the backbone.",
    )

    parser.add_argument(
        "--unfreeze-batch-size",
        type=int,
        default=2,
        help="the batch size for training when unfreeze the backbone.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="""Resume training from from this epoch.
        If it is positive, it will load checkpoint from
        tdnn_lstm_ctc/exp/epoch-{start_epoch-1}.pt
        """,
    )

    # Model Parameters
    # * Use Pretrained Backbone Weights or Not 
    parser.add_argument(
        "--backbone-weights-path",
        type=str,
        default=None,
        help="Path for the pretrained backbone weights file.",
        # When is None, the model doesn't load any pretrained backbone weights file.
        # "download/model_data/backbone_weights/resnet50-19c8e357.pth"
    )

    parser.add_argument(
        "--frozen-weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained"
    )

    # * Backbone
    parser.add_argument(
        "--position-embedding",
        type=str,
        default="sine",
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features"
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        help="Name of the convolutional backbone to use"
    )

    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)"
    )

    # * Transformer
    parser.add_argument(
        "--enc-layers",
        type=int, 
        default=6, 
        help="Number of encoding layers in the transformer"
    )

    parser.add_argument(
        "--dec-layers", 
        type=int,
        default=6, 
        help="Number of decoding layers in the transformer"
    )

    parser.add_argument(
        '--dim-feedforward', 
        type=int,
        default=2048,                        
        help="Intermediate size of the feedforward layers in the transformer blocks"
    )
    
    parser.add_argument(
        '--hidden-dim',
        type=int, 
        default=256, 
        help="Size of the embeddings (dimension of the transformer)"
    )

    parser.add_argument(
        '--dropout', 
        type=float,
        default=0.1, 
        help="Dropout applied in the transformer"
    )

    parser.add_argument(
        '--nheads', 
        type=int,
        default=8, 
        help="Number of attention heads inside the transformer's attentions"
    )

    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of query slots"
    )

    parser.add_argument(
        '--pre-norm', 
        action='store_true'
    )

    # * Segmentation
    parser.add_argument(
        "--masks", 
        action="store_true",
        help="Train segmentation head if the flag is provided"
    )

    # Loss
    parser.add_argument(
        "--no-aux-loss",
        dest="aux_loss",
        action="store_false",  # if write --no-aux-loss, False, or is True
        help="Output auxiliary decoding losses (loss at each layer)"
    )

    # * Matcher
    parser.add_argument(
        "--set-cost-class", 
        type=float,
        default=1, 
        help="Class coefficient in the matching cost"
    )

    parser.add_argument(
        "--set-cost-bbox",
        type=float, 
        default=5, 
        help="L1 box coefficient in the matching cost"
    )

    parser.add_argument(
        "--set-cost-giou", 
        type=float,
        default=2, 
        help="giou box coefficient in the matching cost"
    )

    # * Loss coefficients
    parser.add_argument(
        "--mask-loss-coef",
        type=float, 
        default=1, 
    )

    parser.add_argument(
        "--dice-loss-coef",
        type=float, 
        default=1, 
    )

    parser.add_argument(
        "--bbox-loss-coef",
        type=float, 
        default=5, 
    )
    
    parser.add_argument(
        "--giou-loss-coef", 
        type=float,
        default=2, 
    )

    parser.add_argument(
        "--eos-coef",
        type=float, 
        default=0.1, 
        help="Relative classification weight of the no-object class"
    )

    # * learning rate and related parameters
    parser.add_argument(
        "--lr", 
        type=float,
        default=1e-4,
        help="the learning rate"
    )

    parser.add_argument(
        "--lr-backbone", 
        type=float,
        default=1e-5,
        help="the learning rate for backbone"
    )

    parser.add_argument(
        "--clip-max-norm",
        type=float,
        default=0.1,
        help="the gradient clipping max norm"
    )

    # Optimize strategy
    parser.add_argument(
        "--separate-optimize",
        type=bool,
        default=False,
        help="optimize the backbone and transformer separately with different lr"
    )

    # training environment
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="""the number of workers for training.
        """,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    is saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - exp_dir: It specifies the directory where all training related
                   files, e.g., checkpoints, log, etc, are saved

        - lang_dir: It contains language related input files such as
                    "lexicon.txt"

        - lr: It specifies the initial learning rate

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - weight_decay:  The weight_decay for the optimizer.

        - subsampling_factor:  The subsampling factor for the model.

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval` is 0

        - beam_size: It is used in k2.ctc_loss

        - reduction: It is used in k2.ctc_loss

        - use_double_scores: It is used in k2.ctc_loss
    """
    params = AttributeDict(
        {
            "data_dir": Path("data/voc2007"),
            "lr": 1e-4,
            "weight_decay": 5e-4,
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 10,
            "reset_interval": 200,
            "valid_interval": 1000,
            "save_checkpoint_interval": 5,
            "reduction": "sum",
            "use_double_scores": True,

            "input_shape": [600, 600],
            "backbone": "resnet50", ## vgg
            "pretrained": False,
            "anchors_size": [8, 16, 32],
            "init_lr": 1e-4,
            "optimizer_type": "adam",
            "momentum": 0.9,
            "weight_decay": 0,
            "lr_decay_type": "cos",
            "save_period": 5,
            "eval_flag": True,
            "eval_period": 5,
            "scale": 1,

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
    if params.start_epoch <= 0:
        return

    filename = Path(params.exp_dir) / f"epoch-{params.start_epoch-1}.pt"
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
    filename = Path(params.exp_dir) / f"epoch-{params.cur_epoch}.pt"

    save_checkpoint_impl(
        filename=filename,
        model=model,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = Path(params.exp_dir) / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = Path(params.exp_dir) / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


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
    images, targets = batch[0], batch[1]

    device = model.device
    images = images.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    num_objects = 0
    for target in targets:
        num_objects += target["labels"].size(-1)

    outputs = model(images)

    loss_dict = {}
    losses = 0
    with torch.set_grad_enabled(is_training):
        # Losses: loss_ce, class_error, loss_bbox, loss_giou, cardinality_error
        # loss_ce: classification loss NLL
        # class_error: compute the precision@k for the specified values of k, just for log
        # loss_bbox: the L1 regression loss
        # loss_giou: the giou loss for object detection
        # cardinality_error: compute the cardinality error, the absolute error in the number of 
        # predicted non-empty boxes, not a real loss, just for log 
        # Note: here, we don't talk about segmentation task, so there is no loss_masks

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    assert losses.requires_grad == is_training

    info = MetricsTracker()
    info["num_objects"] = num_objects
    
    for k in loss_dict.keys():
        info[k] = loss_dict[k].detach().cpu().item()

    info["total_sum_loss"] = losses.detach().cpu().item()
    
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

    for batch_idx, batch in enumerate(valid_dl):
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

    loss_value = tot_loss["total_sum_loss"] / tot_loss["num_objects"]

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
        The criterion for detr.
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

    for batch_idx, batch in enumerate(train_dl):
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

    loss_value = tot_loss["total_sum_loss"] / tot_loss["num_objects"]
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

    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    setup_logger(f"{Path(params.exp_dir)}/log/log-train")
    logging.info("Training started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{Path(params.exp_dir)}/tensorboard")
    else:
        tb_writer = None

    num_classes = 20 if args.dataset_file != "coco" else 91
    if args.dataset_file == "coco_panoptic": num_classes = 250    

    model, criterion, postprocessors = build_model(
        args, 
        num_classes = num_classes,
        device = device,
        )

    ## get the number of model's total parameter and the trainable parameter
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total_number of params: {}, trainable_number of params: {}'.format(total_num, trainable_num))

    if params.start_epoch == 0 and params.backbone_weights_path is not None:
        print('Load weights {}.'.format(params.backbone_weights_path))
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(params.backbone_weights_path, map_location = "cpu")
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            # Change k name according your actual case
            k = "backbone.0.body." + k
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        print("Successful Load Key Num:", len(load_key))
        print("Fail To Load Key num:", len(no_load_key))

    checkpoints = load_checkpoint_if_available(params=params, model=model)

    model.to(device)

    if params.freeze_train:
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    if args.separate_optimize and not params.freeze_train:
        print("Optimize the backbone and transformer separately with different lr")
        param_dicts_separate = [
            {
                "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
            },
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    param_dicts = model.parameters()
    if args.separate_optimize and not params.freeze_train:
        param_dicts = param_dicts_separate

    optimizer = optim.AdamW(
        param_dicts,
        lr=params.lr,
        weight_decay=params.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)

    if checkpoints:
        optimizer.load_state_dict(checkpoints["optimizer"])
        scheduler.load_state_dict(checkpoints["scheduler"])

    train_dataset = build_dataset(image_set = "train", args=args)
    valid_dataset = build_dataset(image_set = "val", args=args)

    if world_size > 1:
        sampler_train = DistributedSampler(train_dataset)
        sampler_val = DistributedSampler(valid_dataset, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(valid_dataset)

    batch_size = params.freeze_batch_size if params.freeze_train else params.unfreeze_batch_size

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size, drop_last=True
    )

    train_dl = DataLoader(
        train_dataset, batch_sampler=batch_sampler_train, 
        num_workers = args.num_workers,  
        pin_memory=True, collate_fn=utils.collate_fn)
    valid_dl = DataLoader(
        valid_dataset, sampler=sampler_val,
        batch_size = batch_size, num_workers = args.num_workers,
        pin_memory=True, drop_last=False, collate_fn=utils.collate_fn)

    num_samples = len(train_dataset)
    
    if params.valid_interval >= int(num_samples / batch_size):
        params.valid_interval = int(num_samples / batch_size) - 1

    for epoch in range(params.start_epoch, params.num_epochs):
        
        if epoch >= params.freeze_epoch and params.freeze_train:
            batch_size = params.unfreeze_batch_size

            for param in model.extractor.parameters():
                param.requires_grad = True

            params.lr = 1e-5

            optimizer = optim.AdamW(
                model.parameters(),
                lr=params.lr,
                weight_decay=params.weight_decay,
            )

            batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, batch_size, drop_last=True)

            train_dl = DataLoader(
                train_dataset, batch_sampler=batch_sampler_train, num_workers = args.num_workers,  
                pin_memory=True, collate_fn=utils.collate_fn)
            valid_dl = DataLoader(valid_dataset, sampler=sampler_val, batch_size = batch_size, 
                num_workers = args.num_workers, pin_memory=True, drop_last=False, collate_fn=utils.collate_fn)

            num_samples = len(train_dataset)
    
            if params.valid_interval >= int(num_samples / batch_size):
                params.valid_interval = int(num_samples / batch_size) - 1

        
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

        train_one_epoch(
            params=params,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_dl=train_dl,
            valid_dl=valid_dl,
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

    logging.info("Done!")
    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    args = parser.parse_args()

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()