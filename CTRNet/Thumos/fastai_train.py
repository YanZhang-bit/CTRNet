import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import fastai.train as train
import fastai.basic_data as basic_data
from fastai_callback import Design_tensorboard,Model_save
from matplotlib import pyplot as plot

import pathlib
import torch.nn.parallel

import torch
import torch.nn.parallel
import opts
from gtad_lib.fastai_models import CTRNet
from fastai_dataset import VideoDataSet
from gtad_lib.loss_function import FinalLoss
import random
import numpy as np
seed = random.randint(1,10000)
#seed = 8162
print(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main():
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["output"]):
        os.makedirs(opt["output"])
    model = CTRNet(opt).cuda()
    #model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()

    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=opt["batch_size"], shuffle=False,
                                              num_workers=8, pin_memory=True)
    data_bouch = basic_data.DataBunch(train_dl=train_loader, valid_dl=test_loader)
    learner = train.Learner(data_bouch, model)
    tboard_path = pathlib.Path('/data/zy/project/CTRNet/Thumos/tensorboard/project_5')
    learner.loss_func =FinalLoss(opt)
    learner.path = learner.path / '/data/zy/project/CTRNet/Thumos/'
    learner.model_dir = 'output/'
    learner.wd = opt["weight_decay"]
    learner.callbacks = [Design_tensorboard(base_dir=tboard_path, name='run1')]
    learner.callback_fns = [Model_save]
    learner.fit_one_cycle(15, max_lr=0.001)



if  __name__ == '__main__':
    main()