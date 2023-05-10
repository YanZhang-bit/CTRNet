import math

import numpy as np
import torch
import fastai.callback as callback
import fastai.metrics as metrics
from gtad_lib.loss_function import get_mask
import torch.nn.functional as F
from fastai.basic_train import Learner
from fastai.basic_train import LearnerCallback
from fastai.basic_data import DatasetType, DataBunch
from fastai.torch_core import *
from threading import Thread, Event
from time import sleep
from queue import Queue
import statistics
import torchvision.utils as vutils
from abc import ABC
from tensorboardX import SummaryWriter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


class Design_tensorboard(callback.Callback):
    def __init__(self,  base_dir: Path, name: str,num_batch: int = 100):
        self.base_dir= base_dir
        self.name=name
        self.num_batch = num_batch
        log_dir = base_dir / name
        self.tbwriter = SummaryWriter(str(log_dir),flush_secs=30)
        self.data = None
        self.train_loss=AverageMeter()
        self.val_loss=AverageMeter()
        self.epoch=0
        self.metrics_root = '/metrics/'
    def on_epoch_begin(self, **kwargs:Any) ->None:
        self.train_loss.reset()
        self.val_loss.reset()
    def _write_training_loss(self, iteration: int, epoch_loss: Tensor) -> None:
        "Writes training loss to Tensorboard."
        scalar_value = to_np(epoch_loss)
        tag = self.metrics_root + 'train_loss'
        self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)

    def _write_testing_loss(self, iteration: int, epoch_loss: Tensor) -> None:
        "Writes training loss to Tensorboard."
        scalar_value = to_np(epoch_loss)
        tag = self.metrics_root + 'test_loss'
        self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)


    def _write_scalar(self, name: str, scalar_value, iteration: int) -> None:
        "Writes single scalar value to Tensorboard."
        tag = self.metrics_root + name
        self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)

    # TODO:  Relying on a specific hardcoded start_idx here isn't great.  Is there a better solution?

    def _write_metrics(self, iteration: int, last_metrics: MetricsList) -> None:
        "Writes training metrics to Tensorboard."
        scalar_value = last_metrics[1]
        self._write_scalar(name='recall', scalar_value=scalar_value, iteration=iteration)

    def _write_accuracy(self, iteration: int, last_metrics: MetricsList) -> None:
        "Writes training metrics to Tensorboard."
        scalar_value = last_metrics[2]
        self._write_scalar(name='accuracy', scalar_value=scalar_value, iteration=iteration)

    def _write_val_loss(self, iteration: int, last_metrics: MetricsList) -> None:
        "Writes training metrics to Tensorboard."
        scalar_value = last_metrics[0]
        self._write_scalar(name='val_loss', scalar_value=scalar_value, iteration=iteration)


    def on_batch_end(self, last_loss: Tensor, last_target , iteration: int, train: bool ,**kwargs) -> None:
        "Callback function that writes batch end appropriate data to Tensorboard."

        batch_size=last_target[0][1].size()[0]
        if train:
            self.train_loss.update(last_loss.item(), batch_size)
    # Doing stuff here that requires gradient info, because they get zeroed out afterwards in training loop

    def on_epoch_end(self, last_metrics: MetricsList, **kwargs) -> None:
        "Callback function that writes epoch end appropriate data to Tensorboard."
        train_loss=self.train_loss.avg
        train_loss=tensor(train_loss)
        self.epoch=self.epoch+1
        self._write_training_loss(iteration=self.epoch,epoch_loss=train_loss)
        #self._write_metrics(iteration=self.epoch, last_metrics=last_metrics)
        self._write_val_loss(iteration=self.epoch, last_metrics=last_metrics)




class Model_save(LearnerCallback):
    def __init__(self,learn:Learner):
        super().__init__(learn)
        self.loss=0
        self.epoch=0

    def on_epoch_end(self, last_metrics: MetricsList,**kwargs) -> None:
        "Callback function that writes epoch end appropriate data to Tensorboard."
        self.loss = float(last_metrics[0])
        self.epoch=self.epoch+1
        #scalar_value_ac = float(last_metrics[2])
        if self.epoch%1==0:
            self.learn.save('epoch_%d_model_content'%self.epoch)