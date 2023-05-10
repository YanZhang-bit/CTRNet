import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
from fastai_dataset import VideoDataSet

import json
import torch
import torch.nn.parallel
import torch.optim as optim
import numpy as np
import opts
from models import CTRNet
import pandas as pd
from post_processing import Post_processing
from eval import evaluation_proposal

sys.dont_write_bytecode = True


def Inference(opt):
    model = CTRNet(opt).cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/" + opt["checkpoint"])
    model.load_state_dict(checkpoint['model'])
    # model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)
    tscale = opt["temporal_scale"]
    with torch.no_grad():
        for idx, input_data in test_loader:
            video_name = test_loader.dataset.video_list[idx[0]]
            input_data = input_data.cuda()
            confidence_map, action,start, end = model(input_data)

            # print(start.shape,end.shape,confidence_map.shape)
            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()
            action_scores = action[0].detach().cpu().numpy()
            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

            # 遍历起始分界点与结束分界点的组合
            new_props = []
            for idx in range(tscale):
                a = idx // 2
                for jdx in range(tscale - a):
                    start_index = jdx
                    end_index = jdx + idx + 1
                    if end_index < tscale:
                        xmin = start_index / tscale
                        xmax = end_index / tscale
                        xmin_score = start_scores[start_index]
                        xmax_score = end_scores[end_index]
                        mid_score=action_scores[int((start_index+end_index)/2)]
                        clr_score = clr_confidence[idx, jdx + a]
                        reg_score = reg_confidence[idx, jdx + a]
                        score = xmin_score * xmax_score * clr_score * reg_score
                        new_props.append([xmin, xmax, xmin_score, xmax_score, mid_score,clr_score, reg_score, score])
            new_props = np.stack(new_props)
            #########################################################################

            col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "mid_score","clr_score", "reg_socre", "score"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv("./output/BMN_results/" + video_name + ".csv", index=False)


def main(opt):
    if not os.path.exists("output/BMN_results"):
        os.makedirs("output/BMN_results")
    Inference(opt)

    print("Post processing start")
    Post_processing(opt)
    print("Post processing finished")
    evaluation_proposal(opt)




if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(opt["checkpoint_path"] + "/opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()

    # model = BMN(opt)
    # a = torch.randn(1, 400, 100)
    # b, c = model(a)
    # print(b.shape, c.shape)
    # print(b)
    # print(c)
    main(opt)
