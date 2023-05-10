import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import sys
import math
import numpy as np
import pandas as pd
import torch.nn.parallel

import opts
from models import CTRNet
from fastai_dataset import VideoDataSet


def inferrence(opt):
    model = CTRNet(opt).cuda(0)
    # model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    checkpoint = torch.load(opt['output'] + "/" + opt['checkpoint'])

    model.load_state_dict(checkpoint['model'])
    model.eval()

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation", mode='inference'),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)

    print("Inference start")
    with torch.no_grad():
        for idx, input_data in test_loader:
            video_name = test_loader.dataset.video_list[idx[0]]
            offset = min(test_loader.dataset.data['indices'][idx[0]])
            # print(offset)
            video_name = video_name + '_{}'.format(math.floor(offset / 250))
            input_data = input_data.cuda()

            # forward pass
            confidence_map,action, start, end = model(input_data)

            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()
            action_scores = action[0].detach().cpu().numpy()
            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

            # enumerate sub-graphs as proposals
            new_props = []
            for idx in range(opt["max_duration"]):
                a = idx // 2
                for jdx in range(opt["temporal_scale"]):
                    start_index = jdx
                    end_index = start_index + idx + 1
                    if end_index < opt["temporal_scale"]:
                        xmin = start_index * opt['skip_videoframes'] + offset  # map [0,99] to frames
                        xmax = end_index * opt['skip_videoframes'] + offset
                        xmin_score = start_scores[start_index]
                        xmax_score = end_scores[end_index]
                        mid_score=action_scores[int((start_index+end_index)/2)]
                        clr_score = clr_confidence[idx, jdx + a]  # 64, 128
                        reg_score = reg_confidence[idx, jdx + a]
                        new_props.append([xmin, xmax, xmin_score, xmax_score,mid_score, clr_score, reg_score])
            new_props = np.stack(new_props)

            col_name = ["xmin", "xmax", "xmin_score", "xmax_score","mid_scorce", "clr_score", "reg_socre"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv(opt["output"] + "/results/" + video_name + ".csv", index=False)

    print("Inference finished")


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt['output'] + "/results"):
        os.makedirs(opt['output'] + "/results")
    inferrence(opt)