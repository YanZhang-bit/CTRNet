import os

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import sys
from dataset import VideoDataSet, VideoDataSet_change
from loss_function import bmn_loss_func, get_mask, get_mask_c
import os
import json
import torch
import torch.nn.parallel
import torch.optim as optim
import numpy as np
import opts
from models import BMN, BMN_change, BMN_Ad, BMN_Pac, BMN_Ad_change, BMN_Pac_change, BMN_Pacorg_change
import pandas as pd
from post_processing import BMN_post_processing
from eval import evaluation_proposal

sys.dont_write_bytecode = True


def AP(logits, targets):
    delta = 0.01
    metric = torch.zeros(1).cuda()

    if torch.max(targets) <= 0:
        return metric

    labels_p = (targets == 1)
    fg_logits = logits[labels_p]
    threshold_logit = torch.min(fg_logits) - delta

    ######## Ignore those negative j that satisfy (L_{ij}=0 for all positive i), to accelerate the AP-loss computation.
    valid_labels_n = ((targets == 0) & (logits >= threshold_logit))
    valid_bg_logits = logits[valid_labels_n]
    ########

    fg_num = len(fg_logits)
    prec = torch.zeros(fg_num).cuda()
    order = torch.argsort(fg_logits)
    max_prec = 0

    for ii in order:
        tmp1 = fg_logits - fg_logits[ii]
        tmp1 = (tmp1 >= 0).float()
        tmp2 = valid_bg_logits - fg_logits[ii]
        tmp2 = (tmp2 >= 0).float()
        a = torch.sum(tmp1)
        b = torch.sum(tmp2)
        tmp2 /= (a + b)
        current_prec = a / (a + b)
        if (max_prec <= current_prec):
            max_prec = current_prec
        else:
            tmp2 *= ((1 - max_prec) / (1 - current_prec))
        prec[ii] = max_prec
    fg_num = max(fg_num, 1)
    metric = torch.sum(prec, dim=0, keepdim=True) / fg_num

    return metric


def multi_ap(logits, gt_iou_map, mask):
    socres_map = logits * mask
    iou = 0.9
    labels_b = gt_iou_map.new_ones(gt_iou_map.shape) * -1
    pmask = (gt_iou_map > iou).float() * 2
    nmask = (gt_iou_map <= iou).float()
    nmask = nmask * mask
    labels_b = pmask + nmask + labels_b
    ap = AP(socres_map, labels_b)
    return ap.squeeze()


def train_BMN(data_loader, model, optimizer, epoch, bm_mask):
    model.train()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()
        confidence_map, start, end = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()

    print(
        "BMN training loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1)))


def test_BMN(data_loader, model, epoch, bm_mask):
    model.eval()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    recall_metric = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()

        confidence_map, start, end = model(input_data)
        socrce = confidence_map[:, 0, :, :] * confidence_map[:, 1, :, :] * start[:, :, None] * end[:, None, :]
        recall = multi_ap(socrce, label_confidence, bm_mask.cuda())
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()
        recall_metric += recall.cpu().detach().numpy()

    print(
        "BMN testing loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f,recall_metric: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1),
            recall_metric / (n_iter + 1)))

    return recall_metric / (n_iter + 1)


def BMN_Train(opt):
    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    checkpoint = torch.load(opt["checkpoint_path"] + "/" + opt["checkpoint"])
    print(checkpoint['epoch'])
    # model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt["training_lr"],
                           weight_decay=opt["weight_decay"])

    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=opt["batch_size"], shuffle=False,
                                              num_workers=8, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])
    best_loss = 0
    bm_mask = get_mask(opt["temporal_scale"])
    for epoch in range(opt["train_epochs"]):
        epoch = epoch + 15

        train_BMN(train_loader, model, optimizer, epoch, bm_mask)
        epoch_loss = test_BMN(test_loader, model, epoch, bm_mask)
        state = {'epoch': epoch,
                 'loss': epoch_loss,
                 'state_dict': model.state_dict()}
        torch.save(state, opt["checkpoint_path"] + "/BMN_checkpoint_%d.pth.tar" % epoch)
        if epoch_loss > best_loss:
            best_loss = epoch_loss
            torch.save(state, opt["checkpoint_path"] + "/BMN_best.pth.tar")
        scheduler.step()


def BMN_inference(opt):
    model = BMN_Pac(opt).cuda()
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
            confidence_map, start, end = model(input_data)

            # print(start.shape,end.shape,confidence_map.shape)
            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()
            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

            # 遍历起始分界点与结束分界点的组合
            new_props = []
            for idx in range(tscale):
                for jdx in range(tscale):
                    start_index = idx
                    end_index = jdx + 1
                    if start_index < end_index and end_index < tscale:  # yi qian wei <, wo gai cheng <=,kan yi xiao guo
                        xmin = start_index / tscale
                        xmax = end_index / tscale
                        xmin_score = start_scores[start_index]
                        xmax_score = end_scores[end_index]
                        clr_score = clr_confidence[idx, jdx]
                        reg_score = reg_confidence[idx, jdx]
                        score = xmin_score * xmax_score * clr_score * reg_score
                        new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
            new_props = np.stack(new_props)
            #########################################################################

            col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv("./output/BMN_results/" + video_name + ".csv", index=False)


def BMN_inference_change(opt):
    model = BMN_Pacorg_change(opt).cuda()
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
            confidence_map, start, end = model(input_data)

            # print(start.shape,end.shape,confidence_map.shape)
            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()
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
                        clr_score = clr_confidence[idx, jdx + a]
                        reg_score = reg_confidence[idx, jdx + a]
                        score = xmin_score * xmax_score * clr_score * reg_score
                        new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
            new_props = np.stack(new_props)
            #########################################################################

            col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv("./output/BMN_results/" + video_name + ".csv", index=False)


def main(opt):
    if opt["mode"] == "train":
        BMN_Train(opt)
    elif opt["mode"] == "inference":

        if not os.path.exists("output/BMN_results"):
            os.makedirs("output/BMN_results")
        BMN_inference_change(opt)

        print("Post processing start")
        BMN_post_processing(opt)
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
