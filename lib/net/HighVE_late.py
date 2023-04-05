# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn
import pytorch_lightning as pl
import ipdb

class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1, padding_mode='replicate')
        self.attention = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode='replicate',
            bias=False
        )
        with torch.no_grad():
            self.attention.weight.copy_(torch.zeros_like(self.attention.weight))

    def forward(self, x):
        features = self.conv(x)
        # ipdb.set_trace()
        attention_mask = torch.sigmoid(self.attention(x))
        return features * attention_mask


class BaseNetwork(pl.LightningModule):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        '''
        initializes network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight'
                      ) and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class Residual3D(BaseNetwork):
    def __init__(self, numIn, numOut):
        super(Residual3D, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.with_bias = True
        # self.bn = nn.GroupNorm(4, self.numIn)
        self.bn = nn.BatchNorm3d(self.numIn)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(
            self.numIn,
            self.numOut,
            bias=self.with_bias,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2
        )
        # self.bn1 = nn.GroupNorm(4, self.numOut)
        self.bn1 = nn.BatchNorm3d(self.numOut)
        self.conv2 = nn.Conv3d(
            self.numOut, self.numOut, bias=self.with_bias, kernel_size=3, stride=1, padding=1
        )
        # self.bn2 = nn.GroupNorm(4, self.numOut)
        self.bn2 = nn.BatchNorm3d(self.numOut)
        self.conv3 = nn.Conv3d(
            self.numOut, self.numOut, bias=self.with_bias, kernel_size=3, stride=1, padding=1
        )

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv3d(self.numIn, self.numOut, bias=self.with_bias, kernel_size=1)
        self.init_weights()

    def forward(self, x):
        residual = x
        # out = self.bn(x)
        # out = self.relu(out)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.conv3(out)
        # out = self.relu(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)

        return out + residual


class HighVolumeEncoder(BaseNetwork):
    """CycleGan Encoder"""
    def __init__(self, num_in=3, num_out=32, num_stacks=1):
        super(HighVolumeEncoder, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.num_inter = 7
        self.num_stacks = num_stacks
        self.with_bias = True

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(
            self.num_in,
            self.num_inter,
            bias=self.with_bias,
            kernel_size=5,
            stride=2,
            padding=4,
            dilation=2
        )
        # self.bn1 = nn.GroupNorm(4, self.num_inter)
        self.bn1 = nn.BatchNorm3d(self.num_inter)
        self.SA1 = SelfAttention(self.num_inter, self.num_inter)
        
        self.conv2 = nn.Conv3d(
            self.num_inter,
            self.num_inter,
            bias=self.with_bias, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        # self.bn2 = nn.GroupNorm(4, self.num_out)
        self.bn2 = nn.BatchNorm3d(self.num_inter)
        self.SA2 = SelfAttention(self.num_inter, self.num_inter)
        
        
        self.conv3 = nn.Conv3d(
            self.num_inter,
            self.num_inter,
            bias=self.with_bias, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        # self.bn2 = nn.GroupNorm(4, self.num_out)
        self.bn3 = nn.BatchNorm3d(self.num_inter)
        self.SA3 = SelfAttention(self.num_inter, self.num_inter)
        
        
        self.conv4 = nn.Conv3d(
            self.num_inter+self.num_out,
            self.num_out,
            kernel_size=1,
        )
        # self.bn2 = nn.GroupNorm(4, self.num_out)
        self.bn4 = nn.BatchNorm3d(self.num_out)
        

        self.Upsample = nn.Upsample(scale_factor=4, mode='trilinear',align_corners=True)

        for idx in range(self.num_stacks):
            self.add_module("res" + str(idx), Residual3D(self.num_out, self.num_out))

        self.init_weights()

    def forward(self, x, low_vol_feats=None, intermediate_output=True):
        
        # ipdb.set_trace()
        out = self.conv1(x) # 128
        out = self.bn1(out)
        out = self.relu(out)
        out = self.SA1(out)
        
        # ipdb.set_trace()
        out = self.conv2(out) # 128
        out = self.bn2(out)
        out = self.relu(out)
        out = self.SA2(out)
        
        # ipdb.set_trace()
        out = self.conv3(out) # 128
        out = self.bn3(out)
        out = self.relu(out)
        out = self.SA3(out)
        
        # ipdb.set_trace()
        low_feats = self.Upsample(low_vol_feats) # 32 | 128
        out = torch.cat((out, low_feats), dim=1) # (B, 14, 128, 128, 128)
        
        # ipdb.set_trace()
        out = self.conv4(out) # 128
        out = self.bn4(out)
        out = self.relu(out)

        # ipdb.set_trace()
        out_lst = []
        out_lst.append(out)

        # ipdb.set_trace()
        if intermediate_output:
            return out_lst
        else:
            return [out_lst[-1]]
