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
  python faster_rcnn/train.py \
     --world-size 4 \
     --full-libri 1 \
     --max-duration 300 \
     --num-epochs 20
"""

import os
cwd = os.getcwd()                  ## get the current path

import sys
sys.path.append(cwd)               ## add local to package 

## if there exists error called `wandb: Network error (ConnectionError), entering retry loop.`
## if not, we can set `os.environ["WANDB_MODE"] = "online"`
# os.environ["WANDB_API_KEY"] = "15a220c01c2e84719bcbb7e21e4cdcf553f5530d"
# os.environ["WANDB_MODE"] = "online"

from icefall.checkpoint import load_checkpoint
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    setup_logger,
    str2bool,
)

from local.utils import MetricsTracker
from local.utils import get_classes
from local.dataset_voc2007 import VOC2007Dataset, frcnn_dataset_collate

from model import FasterRCNN
from nets.frcnn_training import (
    FasterRCNNTrainer, 
    get_lr_scheduler,
    set_optimizer_lr, 
    weights_init,
)

import wandb
wandb.login()

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
from torch.utils.data import DataLoader

from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--backbone-weights-path",
        type=str,
        default="download/voc2007_model_data/voc_weights_backbone/resnet50-19c8e357.pth",
        help="Path for the pretrained backbone weights file.",
        # When is "", the model doesn't load any pretrained backbone weights file.
    )

    parser.add_argument(
        "--voc-classes-path",
        type=str,
        default="download/voc2007_model_data/voc_classes.txt",
        help="Path for the voc_classes.txt file.",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="faster_rcnn/exp",
        help="Path for the pretrained model weights file, eg: epoch-0.pt.",
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
        "--wandb",
        type=str2bool,
        default=False,
        help="Should various information be logged in wandb.",
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
            "valid_interval": 50,
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
      batch:
        A batch of data. 
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
    """
    images, boxes, labels = batch[0], batch[1], batch[2]

    device = model.device
    images = images.to(device)

    train_util = FasterRCNNTrainer(model)

    losses = ""
    with torch.set_grad_enabled(is_training):
        rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss = train_util.forward(images, boxes, labels, params.scale)
        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]  ## 4 losses
        losses = losses + [sum(losses)]                                    ## 5 losses

    assert losses[-1].requires_grad == is_training

    info = MetricsTracker()
    info["rpn_loc_loss"] = rpn_loc_loss.detach().cpu().item()
    info["rpn_cls_loss"] = rpn_cls_loss.detach().cpu().item()
    info["roi_loc_loss"] = roi_loc_loss.detach().cpu().item()
    info["roi_cls_loss"] = roi_cls_loss.detach().cpu().item()
    info["total_sum_loss"] = losses[-1].detach().cpu().item()

    return losses, info


def compute_validation_loss(
    params: AttributeDict,
    model: nn.Module,
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
            batch=batch,
            is_training=False,
        )

        assert losses[-1].requires_grad is False

        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(losses[-1].device)

    loss_value = tot_loss["total_sum_loss"]

    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    tb_writer: Optional[SummaryWriter] = None,
    wd_writer: Optional[SummaryWriter] = None,
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
      optimizer:
        The optimizer we are using.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      tb_writer:
        Writer to write log messages to tensorboard.
      wd_writer:
        Writer to write log messages to wandb.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
    """
    model.train()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch[0])

        losses, loss_info = compute_loss(
            params=params,
            model=model,
            batch=batch,
            is_training=True,
        )
        loss = losses[-1]  ## optimize the total sum loss
        
        # summary stats.
        tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0, 2.0)
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
            
            if wd_writer is not None:
                loss_info.write_wandb(
                    wd_writer, "train_current_", params.batch_idx_train
                )
                tot_loss.write_wandb(
                    wd_writer, "train_tot_", params.batch_idx_train
                )

        if batch_idx > 0 and batch_idx % params.valid_interval == 0:
            valid_info = compute_validation_loss(
                params=params,
                model=model,
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

            if wd_writer is not None:
                valid_info.write_wandb(
                    wd_writer,
                    "train_valid_",
                    params.batch_idx_train,
                )

    loss_value = tot_loss["tot_sum_loss"]
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

    setup_logger(f"{Path(params.exp_dir)}/log/log-train")
    logging.info("Training started")
    logging.info(params)

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{Path(params.exp_dir)}/tensorboard")
    else:
        tb_writer = None

    if args.wandb:
        wandb_writer = wandb.init(
            project="faster_rcnn_voc2007",
            entity="luomingshuang",
            job_type="training",
            config={
                "pretrained": params.pretrained,
                "init_lr": params.lr,
                "freeze_train": params.freeze_train,
                "freeze_epoch": params.freeze_epoch,
                "freeze_batch_size": params.freeze_batch_size,
                "start_epoch": params.start_epoch,
                "backbone": params.backbone,
            }
            )
    else:
        wandb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)

    class_names, num_classes = get_classes(params.voc_classes_path)

    model = FasterRCNN(
        num_classes, 
        anchor_scales = params.anchors_size, 
        backbone = params.backbone, 
        pretrained = params.pretrained,
        device = device,
        )

    if params.start_epoch == 0 and params.backbone_weights_path != "":
        print('Load weights {}.'.format(params.backbone_weights_path))
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(params.backbone_weights_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
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
        for param in model.extractor.parameters():
            param.requires_grad = False

    model.freeze_bn()

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=params.lr,
        weight_decay=params.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)

    if checkpoints:
        optimizer.load_state_dict(checkpoints["optimizer"])
        scheduler.load_state_dict(checkpoints["scheduler"])

    train_annotation_path = params.data_dir / "2007_train.txt"
    val_annotation_path = params.data_dir / "2007_val.txt"

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()

    batch_size = params.freeze_batch_size if params.freeze_train else params.unfreeze_batch_size

    train_dataset = VOC2007Dataset(train_lines, params.input_shape, train = True)
    valid_dataset = VOC2007Dataset(val_lines, params.input_shape, train = False)

    train_dl = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = params.num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=frcnn_dataset_collate)
    valid_dl = DataLoader(valid_dataset, shuffle = True, batch_size = batch_size, num_workers = params.num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=frcnn_dataset_collate)

    num_samples = len(train_dataset)
    
    if params.valid_interval >= int(num_samples / batch_size):
        params.valid_interval = int(num_samples / batch_size) - 1

    for epoch in range(params.start_epoch, params.num_epochs):
        
        if epoch >= params.freeze_epoch and params.freeze_train:
            batch_size = params.unfreeze_batch_size

            for param in model.extractor.parameters():
                param.requires_grad = True

            model.freeze_bn()

            params.lr = 1e-5

            optimizer = optim.AdamW(
                model.parameters(),
                lr=params.lr,
                weight_decay=params.weight_decay,
            )

            train_dl = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = params.num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=frcnn_dataset_collate)
            valid_dl = DataLoader(valid_dataset, shuffle = True, batch_size = batch_size, num_workers = params.num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=frcnn_dataset_collate)

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

        if params.wandb:
            wandb.log({"train_lr": scheduler.get_last_lr()[0]}, params.batch_idx_train)
            wandb.log({"train_epoch": epoch}, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            optimizer=optimizer,
            train_dl=train_dl,
            valid_dl=valid_dl,
            tb_writer=tb_writer,
            wd_writer=wandb_writer,
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
