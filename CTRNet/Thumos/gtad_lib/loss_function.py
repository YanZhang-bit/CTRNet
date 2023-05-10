# -*- coding: utf-8 -*-
# loss function is adopted from https://github.com/JJBOY/BMN-Boundary-Matching-Network
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def get_mask(tscale, duration=None, duration_min=None):
    ''' make zeros for invalid anchors'''
    duration = duration or tscale
    mask = []
    for idx in range(tscale):
        a=idx//2
        # for each start localtion, onlly top (tscale - idx) ] are valid
        mask.append([0 for i in range(a)]+[1 for i in range(tscale - idx) ] + [0 for i in range(idx-a)])
    mask = np.array(mask, dtype=np.float32)
    if duration: # chunk mask by max duration
        mask = mask[:duration,:]
    if duration_min: # remove shot actions
        mask[:duration_min] = 0
    return torch.Tensor(mask)



def subgraph_loss_func(pred_bm, gt_iou_map, bm_mask, lambda1=10):
    pred_bm_wcs = pred_bm[:, 0].contiguous()
    pred_bm_mse = pred_bm[:, 1].contiguous()

    wce_loss = pem_reg_loss_func(pred_bm_wcs, gt_iou_map, bm_mask)
    mse_loss = pem_cls_loss_func(pred_bm_mse, gt_iou_map, bm_mask)

    subgraph_loss = wce_loss + lambda1* mse_loss
    return subgraph_loss

def node_loss_func(pred_start, pred_end, gt_start, gt_end):
    def bi_loss(pred_score, gt_label):
        pred_score = pred_score.view(-1)
        gt_label = gt_label.view(-1)
        pmask = (gt_label > 0.5).float()
        num_entries = len(pmask)
        num_positive = torch.sum(pmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 0.000001
        loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon)*(1.0 - pmask)
        loss = -1 * torch.mean(loss_pos + loss_neg)
        return loss

    loss_start = bi_loss(pred_start, gt_start)
    loss_end = bi_loss(pred_end, gt_end)
    loss = loss_start + loss_end
    return loss

def action_node_loss_func(pred_action,pred_start, pred_end, gt_action,gt_start, gt_end):
    def bi_loss(pred_score, gt_label):
        pred_score = pred_score.view(-1)
        gt_label = gt_label.view(-1)
        pmask = (gt_label > 0.5).float()
        num_entries = len(pmask)
        num_positive = torch.sum(pmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 0.000001
        loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon)*(1.0 - pmask)
        loss = -1 * torch.mean(loss_pos + loss_neg)
        return loss

    loss_start = bi_loss(pred_start, gt_start)
    loss_end = bi_loss(pred_end, gt_end)
    loss_action=bi_loss(pred_action,gt_action)
    loss = loss_start + loss_end+loss_action
    return loss
########################################################################################

def bmn_loss_func(pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end, bm_mask):
    pred_bm_reg = pred_bm[:, 0].contiguous()
    pred_bm_cls = pred_bm[:, 1].contiguous()

    gt_iou_map = gt_iou_map * bm_mask

    pem_reg_loss = pem_reg_loss_func(pred_bm_reg, gt_iou_map, bm_mask)
    pem_cls_loss = pem_cls_loss_func(pred_bm_cls, gt_iou_map, bm_mask)
    tem_loss = tem_loss_func(pred_start, pred_end, gt_start, gt_end)

    loss = tem_loss + 10 * pem_reg_loss + pem_cls_loss
    return loss, tem_loss, pem_reg_loss, pem_cls_loss


def tem_loss_func(pred_start, pred_end, gt_start, gt_end):
    def bi_loss(pred_score, gt_label):
        pred_score = pred_score.view(-1)
        gt_label = gt_label.view(-1)
        pmask = (gt_label > 0.5).float()
        num_entries = len(pmask)
        num_positive = torch.sum(pmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 0.000001
        loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon)*(1.0 - pmask)
        loss = -1 * torch.mean(loss_pos + loss_neg)
        return loss

    loss_start = bi_loss(pred_start, gt_start)
    loss_end = bi_loss(pred_end, gt_end)
    loss = loss_start + loss_end
    return loss


def pem_reg_loss_func(pred_score, gt_iou_map, mask):

    u_hmask = (gt_iou_map > 0.7).float()
    u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.3)).float()
    u_lmask = ((gt_iou_map <= 0.3) & (gt_iou_map > 0.)).float()
    u_lmask = u_lmask * mask

    num_h = torch.sum(u_hmask)
    num_m = torch.sum(u_mmask)
    num_l = torch.sum(u_lmask)

    r_m = num_h / num_m
    u_smmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_smmask = u_mmask * u_smmask
    u_smmask = (u_smmask > (1. - r_m)).float()

    r_l = num_h / num_l
    u_slmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_slmask = u_lmask * u_slmask
    u_slmask = (u_slmask > (1. - r_l)).float()

    weights = (u_hmask + u_smmask + u_slmask).detach()
    
    loss = F.mse_loss(pred_score * weights, gt_iou_map * weights)

    loss = 0.5 * torch.sum(loss*torch.ones(*weights.shape).cuda()) / torch.sum(weights)
    if loss != loss:
        print('???')

    return loss


def pem_cls_loss_func(pred_score, gt_iou_map, mask):

    pmask = (gt_iou_map > 0.9).float()
    nmask = (gt_iou_map <= 0.9).float()
    nmask = nmask * mask

    num_positive = 10 + torch.sum(pmask) # in case of nan
    num_entries = 10 + num_positive + torch.sum(nmask)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    epsilon = 0.000001
    loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
    loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * nmask
    loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries

    return loss




class FinalLoss(nn.Module):
    def __init__(self,opt,lambda1=10):
        super(FinalLoss, self).__init__()
        self.lambda1=lambda1
        self.bmn_mask = get_mask(opt["temporal_scale"], opt['max_duration']).cuda()

    def forward(self,outputs,targets):
        confidence_map, action, start, end = outputs[0], outputs[1], outputs[2], outputs[3]
        label_confidence, label_action, label_start, label_end = targets[0], targets[1], targets[2], targets[3]
        pred_bm_mse = confidence_map[:, 0].contiguous()
        pred_bm_wcs = confidence_map[:, 1].contiguous()
        mse_loss = pem_reg_loss_func(pred_bm_mse, label_confidence, self.bmn_mask.cuda())
        wce_loss = pem_cls_loss_func(pred_bm_wcs, label_confidence, self.bmn_mask.cuda())
        subgraph_loss = self.lambda1 * mse_loss + wce_loss
        node_loss = action_node_loss_func(action, start, end, label_action, label_start, label_end)
        loss = subgraph_loss + node_loss
        return loss