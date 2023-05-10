import os
#os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = "2"
from numbers import Number
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from pac import PacConv2d
from adative_diated_conv import _PacConvNd,packernel2d,pacconv2d
from torch.autograd.function import Function, once_differentiable
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
import time


def nd_diated_2col(input_nd, kernel_size, stride=1,rate=2):
    """
    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, *kernel_size, *L_{out})` where
          :math:`L_{out} = floor((L_{in} + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)` for non-transposed
          :math:`L_{out} = (L_{in} - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding` for transposed
    """
    n_dims = len(input_nd.shape[2:])
    stride = (stride,) * n_dims if isinstance(stride, Number) else stride

    (bs, nch), in_sz = input_nd.shape[:2], input_nd.shape[2:]
    out_sz = tuple([((i + 2 * p - d * (k - 1) - 1) // s + 1)
                    for (i, k, d, p, s) in zip(in_sz, (1,kernel_size), (1,1), (0,int((kernel_size-1)/2)), stride)])

    # Use PyINN if possible (about 15% faster) TODO confirm the speed-up

    out_list=[]
    for i in range(input_nd.size()[2]):
        outputs = F.unfold(input_nd[:, :, i, :].unsqueeze(2), (1, kernel_size), max(1, int((i + 1) / rate)),
                           (0, max(1, int((i + 1) / rate)) * int((kernel_size - 1) / 2)), stride)
        out_list.append(outputs)
    output = torch.cat(out_list, dim=2)
    out_shape = (bs, nch) + tuple((1, kernel_size)) + out_sz
    output = output.view(*out_shape).contiguous()

    return output


def nd2col(input_nd, kernel_size, stride=1, padding=0, dilation=1):
    """
    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, *kernel_size, *L_{out})` where
          :math:`L_{out} = floor((L_{in} + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)` for non-transposed
          :math:`L_{out} = (L_{in} - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding` for transposed
    """
    n_dims = len(input_nd.shape[2:])
    kernel_size = (kernel_size,) * n_dims if isinstance(kernel_size, Number) else kernel_size
    stride = (stride,) * n_dims if isinstance(stride, Number) else stride
    padding = (padding,) * n_dims if isinstance(padding, Number) else padding
    dilation = (dilation,) * n_dims if isinstance(dilation, Number) else dilation
    (bs, nch), in_sz = input_nd.shape[:2], input_nd.shape[2:]
    out_sz = tuple([((i + 2 * p - d * (k - 1) - 1) // s + 1)
                    for (i, k, d, p, s) in zip(in_sz, kernel_size, dilation, padding, stride)])
    # Use PyINN if possible (about 15% faster) TODO confirm the speed-up

    output = F.unfold(input_nd, kernel_size, dilation, padding, stride)
    out_shape = (bs, nch) + tuple(kernel_size) + out_sz
    output = output.view(*out_shape).contiguous()
    return output

def kernel_weigth(input_feature,idx,weight_type='counsine',alte=3):
    x1_0 = input_feature[:, :, idx:idx + 1, :, :]
    epsilon=1e-8
    if weight_type=='counsine':
        kernel_weigths = torch.cosine_similarity(x1_0, input_feature, dim=1)
    else:
        if weight_type == 'euclidean':
           x = input_feature - x1_0
           x = x * x
           x = torch.sum(x, dim=1)
        elif weight_type == 'lance':
           x1 = torch.abs(input_feature - x1_0)
           x2 = torch.abs(input_feature + x1_0)
           x = x1 / (x2 + epsilon)
           x = torch.mean(x, dim=1)
        elif weight_type == 'Manhattan':
           x = torch.abs(input_feature - x1_0)
           x = torch.sum(x, dim=1)
        elif weight_type == 'chebyshev':
           x = torch.abs(input_feature - x1_0)
           x = torch.max(x, dim=1)[0]
        elif weight_type == 'Minkowski':
           x = torch.norm(input_feature - x1_0, p=alte, dim=1, keepdim=False)

        kernel_weigths = torch.exp_(x.mul_(-0.5))
    return kernel_weigths

class AdpacConv1d(_PacConvNd):
    r"""
    Args (in addition to those of Conv2d):
        kernel_type (str): 'gaussian' | 'inv_{alpha}_{lambda}[_asym][_fixed]'. Default: 'gaussian'
        smooth_kernel_type (str): 'none' | 'gaussian' | 'average_{sz}' | 'full_{sz}'. Default: 'none'
        normalize_kernel (bool): Default: False
        shared_filters (bool): Default: False
        filler (str): 'uniform'. Default: 'uniform'

    Note:
        - kernel_size only accepts odd numbers
        - padding should not be larger than :math:`dilation * (kernel_size - 1) / 2`
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,rate=2, bias=True,
                 kernel_type='gaussian', smooth_kernel_type='none', normalize_kernel=False, shared_filters=False,
                 filler='uniform', native_impl=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        super(AdpacConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, False, _pair(0), bias,
            False, kernel_type, smooth_kernel_type, False, normalize_kernel, shared_filters, filler)

        self.native_impl = native_impl
        self.rate=rate

    def compute_kernel(self, input_for_kernel, input_mask=None):
        return packernel2d(input_for_kernel, input_mask,
                           kernel_size=self.kernel_size, stride=self.stride, rate=self.rate, kernel_type=self.kernel_type,
                           inv_alpha=self.inv_alpha if hasattr(self, 'inv_alpha') else None,
                           inv_lambda=self.inv_lambda if hasattr(self, 'inv_lambda') else None,
                           channel_wise=False, normalize_kernel=self.normalize_kernel)

    def forward(self, input_2d, input_for_kernel=None, kernel=None, mask=None):
        output_mask = None
        if kernel is None:
            if input_for_kernel is None:
                input_for_kernel=input_2d
            kernel, output_mask = self.compute_kernel(input_for_kernel, mask)

        output = pacconv2d(input_2d, kernel, self.weight, self.bias, self.stride,self.rate,self.shared_filters, self.native_impl)

        return output if output_mask is None else (output, output_mask)

class AdpacConv2d(_PacConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,rate=2, bias=True,
                 kernel_type='gaussian', smooth_kernel_type='none', normalize_kernel=False, shared_filters=False,
                 filler='uniform', native_impl=False):
        kernel_size = _pair(kernel_size)
        row_kernel_size=(1,kernel_size[1])
        clo_kernel_size=(1,kernel_size[0])
        stride = _pair(stride)

        super(AdpacConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,False, _pair(0), bias,
            False, kernel_type, smooth_kernel_type, False, normalize_kernel, shared_filters, filler)

        self.row_cov1d=AdpacConv1d(in_channels, out_channels, row_kernel_size, stride=stride,bias=bias,rate=rate,
                 shared_filters=shared_filters, filler=filler, native_impl=native_impl)
        self.clo_cov1d = AdpacConv1d(in_channels, out_channels, clo_kernel_size, stride=stride,bias=bias,rate=rate,
                 shared_filters=shared_filters, filler=filler, native_impl=native_impl)


    def forward(self, input_2d,input_for_kernel=None):
        if input_for_kernel is None:
            input_for_kernel=input_2d
        row_output=self.row_cov1d(input_2d,input_for_kernel)
        clo_input=input_2d.permute(0,1,3,2).contiguous()
        clo_input_for_kernel=input_for_kernel.permute(0,1,3,2).contiguous()
        clo_output=self.clo_cov1d(clo_input,clo_input_for_kernel).permute(0,1,3,2).contiguous()
        output=row_output+clo_output

        return output

class Content_Relation_Module(nn.Module):
    def __init__(self, in_channels, out_channel1,out_channel2, kernel_size, stride,
                 padding=0, dilation=1,rate=2,bias=False,adative_dilate=False):
        super(Content_Relation_Module, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channel1,kernel_size=1,bias=bias),
            nn.ReLU(inplace=True)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channel2,kernel_size=1,bias=bias),
            nn.ReLU(inplace=True)
        )
        self.stride=stride
        self.kernel_size=_pair(kernel_size)
        self.dilation=dilation
        self.padding=padding
        self.adative_dilate=adative_dilate
        self.rate=rate
    def forward(self,input):
        output1=self.conv1(input)
        bs,k_ch1,x_size,y_size=output1.shape
        output2=self.conv2(input)
        bs, k_ch2, x_size, y_size = output2.shape
        if self.adative_dilate:
            output_cat=torch.cat((output1,output2),dim=1)
            x_feature=nd_diated_2col(output_cat, self.kernel_size[0], stride=self.stride, rate=self.rate).view(bs,k_ch1+k_ch2,-1,*output_cat.shape[-2:])
            x_feature1=x_feature[:,:k_ch1,:,:]
            x_feature2=x_feature[:,k_ch1:,:,:]
            self_idx_x = self.kernel_size[0]// 2
            x_relations=kernel_weigth(x_feature1,self_idx_x)
            x_output = torch.einsum('bcwxy,bwxy->bcxy', (x_feature2, x_relations))
            y_output_cat = output_cat.permute(0, 1, 3, 2).contiguous()
            y_feature=nd_diated_2col(y_output_cat, self.kernel_size[1], stride=self.stride, rate=self.rate).view(bs,k_ch1+k_ch2,-1,*y_output_cat.shape[-2:])
            y_feature1 = y_feature[:, :k_ch1, :, :]
            self_idx_y = self.kernel_size[1] // 2
            y_relations=kernel_weigth(y_feature1,self_idx_y)
            y_feature2 = y_feature[:, k_ch1:, :, :]
            y_output = torch.einsum('bcwxy,bwxy->bcxy', (y_feature2, y_relations)).permute(0,1,3,2)
            output=y_output+x_output-output2
        else:
            feature_map1=nd2col(output1,self.kernel_size,self.stride,self.padding,self.dilation)
            x1 = feature_map1.view(bs, k_ch1, -1, *output1.shape[-2:]).contiguous()
            self_idx1 = self.kernel_size[0] * self.kernel_size[1] // 2
            relations=kernel_weigth(x1,self_idx1)
            feature_map2=nd2col(output2,self.kernel_size,self.stride,self.padding,self.dilation)
            x2 = feature_map2.view(bs, k_ch2, -1, *output1.shape[-2:]).contiguous()
            output = torch.einsum('bcwxy,bwxy->bcxy', (x2, relations))
        return output

'''
class Content_Relation_Module_pac(nn.Module):
    def __init__(self, in_channels, out_channel1,out_channel2, kernel_size, stride=1,
                 padding=0, dilation=1,bias=False):
        super(Content_Relation_Module_pac, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channel1,kernel_size=1,bias=bias),
            nn.ReLU(inplace=True)
        )
        self.conv2=PacConv2d(in_channels=in_channels,out_channels=out_channel2,kernel_size=kernel_size,stride=stride, padding=padding, dilation=dilation,bias=bias)
        self.relu=nn.ReLU(inplace=True)

    def forward(self, input):
        kernel_feature=self.conv1(input)
        output=self.conv2(input,kernel_feature)
        output=self.relu(output)
        return output

class Content_Relation_Module_adpac(nn.Module):
    def __init__(self, in_channels, out_channel1,out_channel2, kernel_size, stride=1,
                rate=1,bias=False):
        super(Content_Relation_Module_adpac, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channel1,kernel_size=1,bias=bias),
            nn.ReLU(inplace=True)
        )
        self.conv2=AdpacConv2d(in_channels=in_channels,out_channels=out_channel2,kernel_size=kernel_size,stride=stride, rate=rate,bias=bias)
        self.relu=nn.ReLU(inplace=True)

    def forward(self, input):
        kernel_feature=self.conv1(input)
        output=self.conv2(input,kernel_feature)
        output=self.relu(output)
        return output
'''
if __name__ == '__main__':
    import opts
    a=torch.tensor(2.0)
    opt = opts.parse_opt()
    opt = vars(opt)
    model=Content_Relation_Module(20,30,40,3,1,1).cuda()
    input=torch.randn(2,20,100,90).cuda()
    parse_time=time.time()
    output=model(input)
    print(time.time()-parse_time)
    print(output.size())#0.7599358558654785
