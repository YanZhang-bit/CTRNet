import math
from numbers import Number
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function, once_differentiable
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
try:
    import pyinn as P

    has_pyinn = True
except ImportError:
    P = None
    has_pyinn = False
    pass

def _neg_idx(idx):
    return None if idx == 0 else -idx


def np_gaussian_2d(width, sigma=-1):
    '''Truncated 2D Gaussian filter'''
    assert width % 2 == 1
    if sigma <= 0:
        sigma = float(width) / 4

    r = np.arange(-(width // 2), (width // 2) + 1, dtype=np.float32)
    gaussian_1d = np.exp(-0.5 * r * r / (sigma * sigma))
    gaussian_2d = gaussian_1d.reshape(-1, 1) * gaussian_1d
    gaussian_2d /= gaussian_2d.sum()

    return gaussian_2d


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

def adconv2d(input,weight,rate=2, bias=None, stride=1,shared_filters=False, native_impl=True):
    kernel_size = tuple(weight.shape[-2:])
    stride = _pair(stride)

    if native_impl:
        # im2col on input
        im_cols = nd_diated_2col(input, kernel_size[1], stride=stride,rate=rate)

        # main computation
        if shared_filters:
            output = torch.einsum('ijklmn,zykl->ijmn', (im_cols, weight))
        else:
            output = torch.einsum('ijklmn,ojkl->iomn', (im_cols , weight))

        if bias is not None:
            output += bias.view(1, -1, 1, 1)
    return output
class _AdConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias,
                 channel_wise,shared_filters, filler):
        super(_AdConvNd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.channel_wise = channel_wise
        self.shared_filters = shared_filters
        self.filler = filler
        if any([k % 2 != 1 for k in kernel_size]):
            raise ValueError('kernel_size only accept odd numbers')
        if shared_filters:
            assert in_channels == out_channels, 'when specifying shared_filters, number of channels should not change'
        if self.filler in {'pool', 'crf_pool'}:
            assert shared_filters
            self.register_buffer('weight', torch.ones(1, 1, *kernel_size))
            if self.filler == 'crf_pool':
                self.weight[(0, 0) + tuple(k // 2 for k in kernel_size)] = 0  # Eq.5, DenseCRF
        elif shared_filters:
            self.weight = Parameter(torch.Tensor(1, 1, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.filler == 'uniform':
            n = self.in_channels
            for k in self.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            if self.shared_filters:
                stdv *= self.in_channels
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        elif self.filler == 'linear':
            effective_kernel_size = tuple(2 * s - 1 for s in self.stride)
            pad = tuple(int((k - ek) // 2) for k, ek in zip(self.kernel_size, effective_kernel_size))
            assert self.transposed and self.in_channels == self.out_channels
            assert all(k >= ek for k, ek in zip(self.kernel_size, effective_kernel_size))
            w = 1.0
            for i, (p, s, k) in enumerate(zip(pad, self.stride, self.kernel_size)):
                d = len(pad) - i - 1
                w = w * (np.array((0.0,) * p + tuple(range(1, s)) + tuple(range(s, 0, -1)) + (0,) * p) / s).reshape(
                    (-1,) + (1,) * d)
                if self.normalize_kernel:
                    w = w * np.array(tuple(((k - j - 1) // s) + (j // s) + 1.0 for j in range(k))).reshape(
                        (-1,) + (1,) * d)
            self.weight.data.fill_(0.0)
            for c in range(1 if self.shared_filters else self.in_channels):
                self.weight.data[c, c, :] = torch.tensor(w)
            if self.bias is not None:
                self.bias.data.fill_(0.0)
        elif self.filler in {'crf', 'crf_perturbed'}:
            assert len(self.kernel_size) == 2 and self.kernel_size[0] == self.kernel_size[1] \
                   and self.in_channels == self.out_channels
            perturb_range = 0.001
            n_classes = self.in_channels
            gauss = np_gaussian_2d(self.kernel_size[0]) * self.kernel_size[0] * self.kernel_size[0]
            gauss[self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
            if self.shared_filters:
                self.weight.data[0, 0, :] = torch.tensor(gauss)
            else:
                compat = 1.0 - np.eye(n_classes, dtype=np.float32)
                self.weight.data[:] = torch.tensor(compat.reshape(n_classes, n_classes, 1, 1) * gauss)
            if self.filler == 'crf_perturbed':
                self.weight.data.add_((torch.rand_like(self.weight.data) - 0.5) * perturb_range)
            if self.bias is not None:
                self.bias.data.fill_(0.0)
        else:
            raise ValueError('Initialization method ({}) not supported.'.format(self.filler))

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', kernel_type={kernel_type}')
        if self.stride != (1,) * len(self.stride):
            s += ', stride={stride}'
        if self.bias is None:
            s += ', bias=False'
        if self.channel_wise:
            s += ', channel_wise=True'
        if self.shared_filters:
            s += ', shared_filters=True'
        return s.format(**self.__dict__)


class AdConv1d(_AdConvNd):
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

    def __init__(self, in_channels, out_channels, kernel_size,rate=2, stride=1,channel_wise=False,bias=True,
                 shared_filters=False, filler='uniform', native_impl=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        super(AdConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, channel_wise, bias, shared_filters, filler)

        self.native_impl = native_impl
        self.rate=rate


    def forward(self, input_2d):

        output = adconv2d(input_2d, self.weight, self.rate,self.bias, self.stride,self.shared_filters)

        return output


class AdConv2d(_AdConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, rate=2,stride=1,channel_wise=False,bias=True,
                 shared_filters=False, filler='uniform', native_impl=False):
        kernel_size = _pair(kernel_size)
        row_kernel_size=(1,kernel_size[1])
        clo_kerner_size=(1,kernel_size[0])
        stride = _pair(stride)
        super(AdConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, channel_wise, bias, shared_filters, filler)

        self.row_cov1d=AdConv1d(in_channels, out_channels, row_kernel_size,rate=rate, stride=1,channel_wise=False,bias=True,
                 shared_filters=False, filler='uniform', native_impl=False)
        self.clo_cov1d = AdConv1d(in_channels, out_channels, clo_kerner_size,rate=rate, stride=1, channel_wise=False, bias=True,
                                  shared_filters=False, filler='uniform', native_impl=False)


    def forward(self, input_2d):

        row_output=self.row_cov1d(input_2d)
        clo_input=input_2d.permute(0,1,3,2).contiguous()
        clo_output=self.clo_cov1d(clo_input).permute(0,1,3,2).contiguous()
        output=row_output+clo_output

        return output

def packernel2d(input, mask=None, kernel_size=0, stride=1,rate=2,kernel_type='gaussian', inv_alpha=None, inv_lambda=None,
                channel_wise=False, normalize_kernel=False):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    output_mask = False if mask is None else True
    norm = None

    if mask is not None and mask.dtype != input.dtype:
        mask = torch.tensor(mask, dtype=input.dtype, device=input.device)


    in_sz = input.shape[-2:]

    if mask is not None or normalize_kernel:
        mask_pattern = input.new_ones(1, 1, *in_sz)
        mask_pattern = nd_diated_2col(mask_pattern, kernel_size, stride=stride,rate=rate)
        if mask is not None:
            mask = nd_diated_2col(mask, kernel_size, stride=stride,rate=rate)
            if not normalize_kernel:
                norm = mask.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) \
                       / mask_pattern.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        else:
            mask = mask_pattern

    bs, k_ch, in_h, in_w = input.shape

    x = nd_diated_2col(input, kernel_size[1], stride=stride,rate=rate)
    x = x.view(bs, k_ch, -1, *x.shape[-2:]).contiguous()

    self_idx = kernel_size[0] * kernel_size[1] // 2
    feat_0 = x[:, :, self_idx:self_idx + 1, :, :]
    x = x - feat_0
    if kernel_type.find('_asym') >= 0:
        x = F.relu(x, inplace=True)
    # x.pow_(2)  # this causes an autograd issue in pytorch>0.4
    x = x * x
    if not channel_wise:
        x = torch.sum(x, dim=1, keepdim=True)
    if kernel_type == 'gaussian':
        x = torch.exp_(x.mul_(-0.5))  # TODO profiling for identifying the culprit of 5x slow down
        # x = torch.exp(-0.5 * x)
    elif kernel_type.startswith('inv_'):
        epsilon = 1e-4
        x = inv_alpha.view(1, -1, 1, 1, 1) \
            + torch.pow(x + epsilon, 0.5 * inv_lambda.view(1, -1, 1, 1, 1))
    else:
        raise ValueError()
    output = x.view(*(x.shape[:2] + tuple(kernel_size) + x.shape[-2:])).contiguous()


    if mask is not None:
        output = output * mask  # avoid numerical issue on masked positions

    if normalize_kernel:
        norm = output.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

    if norm is not None:
        empty_mask = (norm == 0)
        output = output / (norm + torch.tensor(empty_mask, dtype=input.dtype, device=input.device))
        output_mask = (1 - empty_mask) if output_mask else None
    else:
        output_mask = None

    return output, output_mask


def pacconv2d(input, kernel, weight, bias=None, stride=1,rate=2,shared_filters=False,
              native_impl=False):
    kernel_size = tuple(weight.shape[-2:])
    stride = _pair(stride)
    im_cols = nd_diated_2col(input, kernel_size[0], stride=stride,rate=rate)

    # main computation
    if shared_filters:
        output = torch.einsum('ijklmn,zykl->ijmn', (im_cols * kernel, weight))
    else:
        output = torch.einsum('ijklmn,ojkl->iomn', (im_cols * kernel, weight))

    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    return output



class _PacConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, transposed, output_padding, bias,
                 pool_only, kernel_type, smooth_kernel_type,
                 channel_wise, normalize_kernel, shared_filters, filler):
        super(_PacConvNd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.transposed = transposed
        self.output_padding = output_padding
        self.pool_only = pool_only
        self.kernel_type = kernel_type
        self.smooth_kernel_type = smooth_kernel_type
        self.channel_wise = channel_wise
        self.normalize_kernel = normalize_kernel
        self.shared_filters = shared_filters
        self.filler = filler
        if any([k % 2 != 1 for k in kernel_size]):
            raise ValueError('kernel_size only accept odd numbers')
        if smooth_kernel_type.find('_') >= 0 and int(smooth_kernel_type[smooth_kernel_type.rfind('_') + 1:]) % 2 != 1:
            raise ValueError('smooth_kernel_type only accept kernels of odd widths')
        if shared_filters:
            assert in_channels == out_channels, 'when specifying shared_filters, number of channels should not change'
        if not pool_only:
            if self.filler in {'pool', 'crf_pool'}:
                assert shared_filters
                self.register_buffer('weight', torch.ones(1, 1, *kernel_size))
                if self.filler == 'crf_pool':
                    self.weight[(0, 0) + tuple(k // 2 for k in kernel_size)] = 0  # Eq.5, DenseCRF
            elif shared_filters:
                self.weight = Parameter(torch.Tensor(1, 1, *kernel_size))
            elif transposed:
                self.weight = Parameter(torch.Tensor(in_channels, out_channels, *kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            if bias:
                self.bias = Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)
        if kernel_type.startswith('inv_'):
            self.inv_alpha_init = float(kernel_type.split('_')[1])
            self.inv_lambda_init = float(kernel_type.split('_')[2])
            if self.channel_wise and kernel_type.find('_fixed') < 0:
                if out_channels <= 0:
                    raise ValueError('out_channels needed for channel_wise {}'.format(kernel_type))
                inv_alpha = self.inv_alpha_init * torch.ones(out_channels)
                inv_lambda = self.inv_lambda_init * torch.ones(out_channels)
            else:
                inv_alpha = torch.tensor(float(self.inv_alpha_init))
                inv_lambda = torch.tensor(float(self.inv_lambda_init))
            if kernel_type.find('_fixed') < 0:
                self.register_parameter('inv_alpha', Parameter(inv_alpha))
                self.register_parameter('inv_lambda', Parameter(inv_lambda))
            else:
                self.register_buffer('inv_alpha', inv_alpha)
                self.register_buffer('inv_lambda', inv_lambda)
        elif kernel_type != 'gaussian':
            raise ValueError('kernel_type set to invalid value ({})'.format(kernel_type))
        if smooth_kernel_type.startswith('full_'):
            smooth_kernel_size = int(smooth_kernel_type.split('_')[-1])
            self.smooth_kernel = Parameter(torch.Tensor(1, 1, *repeat(smooth_kernel_size, len(kernel_size))))
        elif smooth_kernel_type == 'gaussian':
            smooth_1d = torch.tensor([.25, .5, .25])
            smooth_kernel = smooth_1d
            for d in range(1, len(kernel_size)):
                smooth_kernel = smooth_kernel * smooth_1d.view(-1, *repeat(1, d))
            self.register_buffer('smooth_kernel', smooth_kernel.unsqueeze(0).unsqueeze(0))
        elif smooth_kernel_type.startswith('average_'):
            smooth_kernel_size = int(smooth_kernel_type.split('_')[-1])
            smooth_1d = torch.tensor((1.0 / smooth_kernel_size,) * smooth_kernel_size)
            smooth_kernel = smooth_1d
            for d in range(1, len(kernel_size)):
                smooth_kernel = smooth_kernel * smooth_1d.view(-1, *repeat(1, d))
            self.register_buffer('smooth_kernel', smooth_kernel.unsqueeze(0).unsqueeze(0))
        elif smooth_kernel_type != 'none':
            raise ValueError('smooth_kernel_type set to invalid value ({})'.format(smooth_kernel_type))

        self.reset_parameters()

    def reset_parameters(self):
        if not (self.pool_only or self.filler in {'pool', 'crf_pool'}):
            if self.filler == 'uniform':
                n = self.in_channels
                for k in self.kernel_size:
                    n *= k
                stdv = 1. / math.sqrt(n)
                if self.shared_filters:
                    stdv *= self.in_channels
                self.weight.data.uniform_(-stdv, stdv)
                if self.bias is not None:
                    self.bias.data.uniform_(-stdv, stdv)
            elif self.filler == 'linear':
                effective_kernel_size = tuple(2 * s - 1 for s in self.stride)
                pad = tuple(int((k - ek) // 2) for k, ek in zip(self.kernel_size, effective_kernel_size))
                assert self.transposed and self.in_channels == self.out_channels
                assert all(k >= ek for k, ek in zip(self.kernel_size, effective_kernel_size))
                w = 1.0
                for i, (p, s, k) in enumerate(zip(pad, self.stride, self.kernel_size)):
                    d = len(pad) - i - 1
                    w = w * (np.array((0.0,) * p + tuple(range(1, s)) + tuple(range(s, 0, -1)) + (0,) * p) / s).reshape(
                        (-1,) + (1,) * d)
                    if self.normalize_kernel:
                        w = w * np.array(tuple(((k - j - 1) // s) + (j // s) + 1.0 for j in range(k))).reshape(
                            (-1,) + (1,) * d)
                self.weight.data.fill_(0.0)
                for c in range(1 if self.shared_filters else self.in_channels):
                    self.weight.data[c, c, :] = torch.tensor(w)
                if self.bias is not None:
                    self.bias.data.fill_(0.0)
            elif self.filler in {'crf', 'crf_perturbed'}:
                assert len(self.kernel_size) == 2 and self.kernel_size[0] == self.kernel_size[1] \
                       and self.in_channels == self.out_channels
                perturb_range = 0.001
                n_classes = self.in_channels
                gauss = np_gaussian_2d(self.kernel_size[0]) * self.kernel_size[0] * self.kernel_size[0]
                gauss[self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
                if self.shared_filters:
                    self.weight.data[0, 0, :] = torch.tensor(gauss)
                else:
                    compat = 1.0 - np.eye(n_classes, dtype=np.float32)
                    self.weight.data[:] = torch.tensor(compat.reshape(n_classes, n_classes, 1, 1) * gauss)
                if self.filler == 'crf_perturbed':
                    self.weight.data.add_((torch.rand_like(self.weight.data) - 0.5) * perturb_range)
                if self.bias is not None:
                    self.bias.data.fill_(0.0)
            else:
                raise ValueError('Initialization method ({}) not supported.'.format(self.filler))
        if hasattr(self, 'inv_alpha') and isinstance(self.inv_alpha, Parameter):
            self.inv_alpha.data.fill_(self.inv_alpha_init)
            self.inv_lambda.data.fill_(self.inv_lambda_init)
        if hasattr(self, 'smooth_kernel') and isinstance(self.smooth_kernel, Parameter):
            self.smooth_kernel.data.fill_(1.0 / np.multiply.reduce(self.smooth_kernel.shape))

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', kernel_type={kernel_type}')
        if self.stride != (1,) * len(self.stride):
            s += ', stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.bias is None:
            s += ', bias=False'
        if self.smooth_kernel_type != 'none':
            s += ', smooth_kernel_type={smooth_kernel_type}'
        if self.channel_wise:
            s += ', channel_wise=True'
        if self.normalize_kernel:
            s += ', normalize_kernel=True'
        if self.shared_filters:
            s += ', shared_filters=True'
        return s.format(**self.__dict__)


class PacConv1d(_PacConvNd):
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
        super(PacConv1d, self).__init__(
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

    def forward(self, input_2d, input_for_kernel, kernel=None, mask=None):
        output_mask = None
        if kernel is None:
            kernel, output_mask = self.compute_kernel(input_for_kernel, mask)

        output = pacconv2d(input_2d, kernel, self.weight, self.bias, self.stride,self.rate,self.shared_filters, self.native_impl)

        return output if output_mask is None else (output, output_mask)

class PacConv2d(_PacConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,rate=2, bias=True,
                 kernel_type='gaussian', smooth_kernel_type='none', normalize_kernel=False, shared_filters=False,
                 filler='uniform', native_impl=False):
        kernel_size = _pair(kernel_size)
        row_kernel_size=(1,kernel_size[1])
        clo_kernel_size=(1,kernel_size[0])
        stride = _pair(stride)

        super(PacConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,False, _pair(0), bias,
            False, kernel_type, smooth_kernel_type, False, normalize_kernel, shared_filters, filler)

        self.row_cov1d=PacConv1d(in_channels, out_channels, row_kernel_size, stride=stride,bias=bias,rate=rate,
                 shared_filters=shared_filters, filler=filler, native_impl=native_impl)
        self.clo_cov1d = PacConv1d(in_channels, out_channels, clo_kernel_size, stride=stride,bias=bias,rate=rate,
                 shared_filters=shared_filters, filler=filler, native_impl=native_impl)


    def forward(self, input_2d):

        row_output=self.row_cov1d(input_2d,input_2d)
        clo_input=input_2d.permute(0,1,3,2).contiguous()
        clo_output=self.clo_cov1d(clo_input,clo_input).permute(0,1,3,2).contiguous()
        output=row_output+clo_output

        return output


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

        self.row_cov1d=PacConv1d(in_channels, out_channels, row_kernel_size, stride=stride,bias=bias,rate=rate,
                 shared_filters=shared_filters, filler=filler, native_impl=native_impl)
        self.clo_cov1d = PacConv1d(in_channels, out_channels, clo_kernel_size, stride=stride,bias=bias,rate=rate,
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