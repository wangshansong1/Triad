# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable
from copy import deepcopy
import torch

import util.misc as misc
import util.lr_sched as lr_sched
torch.set_printoptions(precision=6)


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    """
    Train the model for one epoch during pretraining.

    This function iterates over the provided data_loader for a single epoch, performing
    training with gradient accumulation and mixed-precision support via torch.cuda.amp.
    The learning rate is adjusted on a per-iteration basis, and training metrics such as
    loss and learning rate are logged using a MetricLogger. Optionally, metrics can also
    be recorded using a TensorBoard SummaryWriter.

    Parameters:
        model (torch.nn.Module): The model to train.
        data_loader (Iterable): An iterable over the training data; expected to yield
                                  (samples, _) tuples.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        device (torch.device): The device on which to perform training (e.g., 'cuda' or 'cpu').
        epoch (int): The current epoch number (used for logging and learning rate scheduling).
        loss_scaler: A utility for scaling the loss for mixed-precision training.
        log_writer: (Optional) A TensorBoard SummaryWriter for logging training metrics.
        args: A namespace containing additional training parameters. Must include at least:
              - accum_iter: Number of iterations to accumulate gradients.
              - mask_ratio: The ratio of patches to mask in the model's forward pass.

    Returns:
        dict: A dictionary containing the global average of the tracked metrics (e.g., loss, lr)
              over the epoch.

    Raises:
        SystemExit: If the loss is not finite.
    """

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, text, mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        text = text.to(device, non_blocking=True)

        samples_copy = deepcopy(samples)
        samples_copy = samples_copy.cuda(non_blocking=True)
        mask_copy = deepcopy(mask)
        mask_copy = mask_copy.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():

            loss = model(samples_copy, mask_copy, samples_copy, text)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {:.8f}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
