
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import fastai.train as train
import fastai.basic_data as basic_data
import sys
from fastai_dataset import VideoDataSet
from loss_function import FinalLoss
from fastai_callback import Design_tensorboard,Metric_Recall,Metric_action_amp,Metric_start_amp,Metric_end_amp
from fastai_callback import Model_save as Model_save

import json
import torch
import pathlib
import torch.nn.parallel
import numpy as np
import opts
from fastai_model import CTRNet

import random
seed = random.randint(1,10000)
print(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main():
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(opt["checkpoint_path"] + "/opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()
    if opt["mode"] == "train":
        model = CTRNet(opt)
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
        train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                                   batch_size=opt["batch_size"], shuffle=True,
                                                   num_workers=8, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                                  batch_size=opt["batch_size"], shuffle=False,
                                                  num_workers=8, pin_memory=True)
        data_bouch = basic_data.DataBunch(train_dl=train_loader, valid_dl=test_loader)
        learner = train.Learner(data_bouch, model)
        bmn_loss=FinalLoss(opt)
        tboard_path = pathlib.Path('/data/zy/project/CTRNet/ActivityNet/tensorboard/project_1')
        learner.loss_func = bmn_loss
        learner.metrics=[Metric_Recall(opt)]
        learner.path = learner.path / '/data/zy/project/CTRNet/ActivityNet/'
        learner.model_dir = 'model/'
        learner.metrics = [Metric_action_amp(), Metric_start_amp(),
                           Metric_end_amp()]
        learner.wd = opt["weight_decay"]
        learner.callbacks = [Design_tensorboard(base_dir=tboard_path, name='run1')]
        learner.callback_fns = [Model_save]
        learner.fit_one_cycle(15, max_lr=0.0001)

if  __name__ == '__main__':
    main()