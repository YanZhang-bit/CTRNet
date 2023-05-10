import math
from numbers import Number
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
#from pac import PacConv2d

from content_relation_model import Content_Relation_Module
from temporal_relation_model import AdConv2d


class Content_temporal_module(nn.Module):
    def __init__(self, in_channels, out_channel1, out_channel2,kernel_size,structure=1,stride=1,padding=0, dilation=1,rate=2, bias=True,
                 adative_dilate=True):
        super(Content_temporal_module, self).__init__()
        self.content_module=Content_Relation_Module(in_channels=in_channels,out_channel1=out_channel1,
                                                   out_channel2=out_channel2,kernel_size=kernel_size,
                                                   stride=stride,padding=padding,dilation=dilation,
                                                   rate=rate,bias=bias,adative_dilate=adative_dilate)


        self.structure=structure
        self.in_channels=in_channels
        if structure>5 and structure<9:
            self.tem_in_channels=out_channel2
        else:
            self.tem_in_channels=in_channels

        if adative_dilate:
            self.temporal_module=AdConv2d(in_channels=self.tem_in_channels,out_channels=out_channel2,
                                          kernel_size=kernel_size,rate=rate,stride=stride,bias=bias)
        else:
            self.temporal_module=nn.Conv2d(in_channels=self.tem_in_channels,out_channels=out_channel2,
                                           kernel_size=kernel_size,stride=stride,
                                           padding=padding,dilation=dilation,bias=bias)
        self.out_channels=out_channel2
        if self.in_channels!=self.out_channels:
            self.channel_map=nn.Conv2d(self.out_channels,self.in_channels,1)
        if self.structure==4 or self.structure==5:
            self.concate_map=nn.Conv2d(self.out_channels*2,self.in_channels,1)


    def forward(self,input):
        if self.structure==1:
           tem_output=self.temporal_module(input)
           output=tem_output
           if self.in_channels!=self.out_channels:
               output=self.channel_map(output)
           output=output
        elif self.structure==2:
            cont_output = self.content_module(input)
            tem_output = self.temporal_module(input)
            output = cont_output + tem_output
            if self.in_channels != self.out_channels:
                output = self.channel_map(output)
        elif self.structure==3:
            cont_output = self.content_module(input)
            output = cont_output
            if self.in_channels != self.out_channels:
                output = self.channel_map(output)
        elif self.structure==4:
            cont_output = self.content_module(input)
            tem_output = self.temporal_module(input)
            output =torch.cat((cont_output,tem_output),dim=1)
            output=self.concate_map(output)
        elif self.structure==5:
            cont_output = self.content_module(input)
            tem_output = self.temporal_module(input)
            output =torch.cat((cont_output,tem_output),dim=1)
            output=self.concate_map(output)
            output=input+output
        elif self.structure==6:
            output = self.content_module(input)
            output = self.temporal_module(output)
            if self.in_channels != self.out_channels:
                output = self.channel_map(output)
        elif self.structure==7:
            output = self.content_module(input)
            if self.in_channels != self.out_channels:
                output = self.channel_map(output)
        elif self.structure == 8:
            con_output = self.content_module(input)
            tem_output = self.temporal_module(con_output)
            output = con_output + tem_output
            if self.in_channels != self.out_channels:
                output = self.channel_map(output)
            output=input+output

        return output

'''
class Contentpac_temporal_module(nn.Module):
    def __init__(self, in_channels, out_channel1, out_channel2, kernel_size, structure=1, stride=1, padding=0,
                 dilation=1, rate=2, bias=True,
                 adative_dilate=True):
        super(Contentpac_temporal_module, self).__init__()

        if adative_dilate:
            self.temporal_module = AdConv2d(in_channels=in_channels, out_channels=out_channel2,
                                            kernel_size=kernel_size, rate=rate, stride=stride, bias=bias)
            self.content_module = Content_Relation_Module_adpac(in_channels=in_channels, out_channel1=out_channel1,
                                                          out_channel2=out_channel2, kernel_size=kernel_size,
                                                          stride=stride,rate=rate, bias=bias)
        else:
            self.temporal_module = nn.Conv2d(in_channels=in_channels, out_channels=out_channel2,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, dilation=dilation, bias=bias)
            self.content_module = Content_Relation_Module_pac(in_channels=in_channels, out_channel1=out_channel1,
                                                                out_channel2=out_channel2, kernel_size=kernel_size,
                                                                stride=stride, padding=padding, dilation=dilation,
                                                                bias=bias)

        self.structure = structure
        self.in_channels = in_channels
        self.out_channels = out_channel2
        if self.in_channels != self.out_channels:
            self.channel_map = nn.Conv2d(self.out_channels, self.in_channels, 1)

    def forward(self, input):
        if self.structure == 1:
            cont_output = self.content_module(input)
            tem_output = self.temporal_module(input)
            output = cont_output + tem_output
            if self.in_channels != self.out_channels:
                output = self.channel_map(output)
            output = output + input
        elif self.structure == 2:
            cont_output = self.content_module(input)
            tem_output = self.temporal_module(input)
            output = cont_output + tem_output
            if self.in_channels != self.out_channels:
                output = self.channel_map(output)
        elif self.structure == 3:
            cont_output = self.content_module(input)
            output = cont_output
            if self.in_channels != self.out_channels:
                output = self.channel_map(output)

        return output


class Attention_Conv_Fusion_module(nn.Module):
    def __init__(self, in_channels, out_channel1, out_channel2, kernel_size, stride=1, padding=1, dilation=1, bias=False):
        super(Attention_Conv_Fusion_module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel1, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True)
        )
        self.conv2 = PacConv2d(in_channels=in_channels, out_channels=out_channel2, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, input):
        kernel_feature = self.conv1(input)
        output = self.conv2(input, kernel_feature)
        return output
'''