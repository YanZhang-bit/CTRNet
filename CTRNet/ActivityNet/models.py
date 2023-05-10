# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
from fusion_model import Content_temporal_module




class CTRNet(nn.Module):  # liujiexi
    def __init__(self, opt):
        super(CTRNet, self).__init__()
        self.tscale = opt["temporal_scale"]
        self.prop_boundary_ratio = opt["prop_boundary_ratio"]
        self.num_sample = opt["num_sample"]
        self.num_sample_perbin = opt["num_sample_perbin"]
        self.feat_dim = opt["feat_dim"]

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512
        self.content_kernel_dim = 64

        self._get_interp1d_mask()

        # Base Module
        self.x_1d_b1 = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_3d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)
        self.x_1d_b2 = nn.Conv1d(self.hidden_dim_3d, self.hidden_dim_3d, kernel_size=3, padding=1)

        self.x_1d_b3 = nn.LSTM(input_size=self.hidden_dim_3d, hidden_size=self.hidden_dim_3d,
                               num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)

        # Proposal Evaluation Module
        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.hidden_dim_3d * 2, self.hidden_dim_1d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1),
                      stride=(self.num_sample, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            Content_temporal_module(in_channels=self.hidden_dim_2d, out_channel1=self.content_kernel_dim,
                                    out_channel2=self.hidden_dim_2d, kernel_size=5, structure=2,
                                    padding=2, dilation=2, rate=4,
                                    adative_dilate=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
            nn.Sigmoid()
        )
        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.hidden_dim_3d * 2, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.hidden_dim_3d * 2, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.x_1d_a = nn.Sequential(
            nn.Conv1d(self.hidden_dim_3d * 2, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        base_feature1 = self.x_1d_b1(x)  # (batch_size,chinnal,t)
        base_feature = self.x_1d_b2(base_feature1)
        base_feature = base_feature + base_feature1
        base_feature = self.relu(base_feature)
        base_feature = base_feature.permute(0, 2, 1)
        self.x_1d_b3.flatten_parameters()
        base_feature, (_, _) = self.x_1d_b3(base_feature)
        base_feature = base_feature.permute(0, 2, 1)
        start = self.x_1d_s(base_feature).squeeze(1)
        end = self.x_1d_e(base_feature).squeeze(1)
        action = self.x_1d_a(base_feature).squeeze(1)
        confidence_map = self.x_1d_p(base_feature)
        confidence_map = self._boundary_matching_layer(confidence_map)
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        confidence_map = self.x_2d_p(confidence_map)
        return confidence_map, action, start, end

    def _boundary_matching_layer(self, x):
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0], input_size[1], self.num_sample, self.tscale,
                                                        self.tscale)
        return out

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for duration in range(self.tscale):
            mask_mat_vector = []
            a = duration // 2
            for i in range(a):
                mask_mat_vector.append(np.zeros([self.tscale, self.num_sample]))
            for start_index in range(self.tscale - a):
                if start_index + duration + 1 < self.tscale:
                    p_xmin = start_index
                    p_xmax = start_index + duration + 1
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=2)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)

