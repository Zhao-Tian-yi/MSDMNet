#!/usr/bin/env python
# coding=utf-8
'''
Author: ymx & zty
Date: 2022-10-10
LastEditTime: 2023-01-06
Description: Detailed Memory: Learning to Pan-sharpening with Memories of Spatial Dictionary
batch_size = 16, MAE & KL, Adam, weight_decay = 0.0001, PAN & GT: size = 64, LRMS: size =16, 200 epoch, decay 50, x0.5
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from base_net import *
from torchvision.transforms import *
import torch.nn.functional as F
import numpy as np

class Bottleneck(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf//4, 1, 1, 0),
            nn.PReLU(),
            nn.Conv2d(nf//4, nf//4, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(nf//4, nf, 1, 1, 0),
            )

    def forward(self, x):
        return x + self.conv(x)


class BottleneckGroup(nn.Module):
    def __init__(self, nf, blocks):
        super().__init__()
        group = [Bottleneck(nf) for _ in range(blocks)]
        group += [nn.Conv2d(nf, nf, 1, 1, 0)]
        self.group = nn.Sequential(*group)

    def forward(self, x):
        return x + self.group(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return torch.sum(self.sigmoid(out)*x,dim=1,keepdim=True)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3,5, 7), 'kernel size must be 3 or 7'
        if kernel_size == 7:
            padding = 3
        elif kernel_size == 5:
            padding = 2
        elif kernel_size == 3:
            padding = 1

        self.conv1 = nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        min_out, _ = torch.min(x, dim=1 ,keepdim=True)
        x = torch.cat([avg_out, max_out , min_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DRBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 3, 1, 1),
            nn.BatchNorm2d(mid_c),
        )
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

        self.conv1 = nn.Sequential(
            nn.Conv2d(mid_c, out_c//2, 3, 1, 1),
            nn.BatchNorm2d(out_c//2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_c, out_c//2, 3, 1, 1),
            nn.BatchNorm2d(out_c//2),
        )

    def forward(self, x):
        a1 = self.head(x)
        a21 = self.prelu1(a1)
        a22 = self.prelu2(-a1)
        a31 = self.conv1(a21)
        a32 = self.conv2(a22)
        out = torch.cat([a31,-a32],dim=1)
        return out


class DRN(nn.Module):
    def __init__(self, nfs, in_c=1,out_c=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

        self.initial = nn.Conv2d(in_c, nfs[0], 3, 1, 1)

        self.conv0_0 = DRBlock(in_c, nfs[0], nfs[0])
        self.conv1_0 = DRBlock(nfs[0], nfs[1], nfs[1])
        self.conv2_0 = DRBlock(nfs[1], nfs[2], nfs[2])

        self.conv0_1 = DRBlock(nfs[0]+nfs[1], nfs[0], nfs[0])
        self.conv1_1 = DRBlock(nfs[1]+nfs[2], nfs[1], nfs[1])

        self.conv0_2 = DRBlock(nfs[0]*2+nfs[1], nfs[0], nfs[0])

        self.final = nn.Conv2d(nfs[0]+nfs[0], out_c, 3 , 1, 1)

    def forward(self, x):
        xini = self.initial(x)

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        return self.final(torch.cat([x0_2, xini], 1))

class Memory_Block(nn.Module):
    def __init__(self, n_atoms=64, scale=4,ms_band_num=4,batch_size=16):
        super().__init__()
        self.scale=scale
        self.n_atoms = n_atoms
        self.units = nn.Embedding(batch_size*n_atoms,scale**2)
        self.band=ms_band_num
        self.batch_size = batch_size
        # self.code = nn.Sequential(
        #     nn.Conv2d(n_atoms, n_atoms, scale, scale, 0, groups=n_atoms),
        #     nn.PReLU(),
        #     nn.Conv2d(n_atoms,n_atoms, 1, 1, 0)
        # )
    def forward(self):
        '''
          x: (b, c, h, w)
          embed: (k, c)
        '''
        # B,C,h,w = x.size()
        m = self.units.weight  # (k, n)
        m_ = m.reshape((self.batch_size,self.n_atoms,self.scale,self.scale))
        # x_=x.reshape((1,-1,4,4))
        return m_

class MSDMNet(nn.Module):
    def __init__(self, num_channels,batch_size,n_atoms=64):
        super(MSDMNet, self).__init__()

        nfs     = [n_atoms, n_atoms*2, n_atoms*4 ]
        self.batch_size=batch_size
        self.n_atoms=n_atoms
        self.head = nn.Sequential(
            nn.Conv2d(num_channels, nfs[0]//2, 1, 1, 0),
            BottleneckGroup(nfs[0]//2, blocks=3)
        )
        out_channels = 4
        n_resblocks = 11

        res_block_s1 = [
            ConvBlock(nfs[0]//2, 32, 3, 1, 1, activation='prelu', norm=None, bias=False),
        ]
        for i in range(n_resblocks):
            res_block_s1.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        res_block_s1.append(Upsampler(2, 32, activation='prelu'))
        res_block_s1.append(ConvBlock(32, self.n_atoms, 3, 1, 1, activation='prelu', norm=None, bias=False))
        self.res_block_s1 = nn.Sequential(*res_block_s1)

        res_block_s2 = [
            ConvBlock(self.n_atoms, 32, 3, 1, 1, activation='prelu', norm=None, bias=False),
        ]
        for i in range(n_resblocks):
            res_block_s2.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        res_block_s2.append(ConvBlock(32, 4, 3, 1, 1, activation='prelu', norm=None, bias=False))
        self.res_block_s2 = nn.Sequential(*res_block_s2)

        res_block_s3 = [
            ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias=False),
        ]
        for i in range(n_resblocks):
            res_block_s3.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        res_block_s3.append(Upsampler(2, 32, activation='prelu'))
        res_block_s3.append(ConvBlock(32, self.n_atoms, 3, 1, 1, activation='prelu', norm=None, bias=False))
        self.res_block_s3 = nn.Sequential(*res_block_s3)

        res_block_s4 = [
            ConvBlock(self.n_atoms, 32, 3, 1, 1, activation='prelu', norm=None, bias=False),
        ]
        for i in range(n_resblocks):
            res_block_s4.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        res_block_s4.append(ConvBlock(32, 4, 3, 1, 1, activation='prelu', norm=None, bias=False))
        self.res_block_s4 = nn.Sequential(*res_block_s4)

        self.Memory_2 = Memory_Block(scale=2)
        self.Memory_4 = Memory_Block(scale=4)

        self.cattention_2 = ChannelAttention(in_channel=self.n_atoms,ratio=8)
        self.cattention_4 = ChannelAttention(in_channel=self.n_atoms,ratio=8)

        self.Unet_hp2 = DRN(nfs, in_c=1, out_c=1)
        self.Unet_hp4 = DRN(nfs, in_c=1, out_c=1)


        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()

        # self.m1_conv1_1= ConvBlock(4 ,1, kernel_size=1, stride=1, padding=0, bias=True, activation='prelu', norm=None, pad_model=None)
        # self.m2_cov_1_1= ConvBlock(4 ,1, kernel_size=1, stride=1, padding=0, bias=True, activation='prelu', norm=None, pad_model=None)

    def truncated_linear_stretch(image, truncated_value, max_out=255, min_out=0):
        def gray_process(gray):
            truncated_down = np.percentile(gray, truncated_value)
            truncated_up = np.percentile(gray, 100 - truncated_value)
            gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
            gray[gray < min_out] = min_out
            gray[gray > max_out] = max_out
            if (max_out <= 255):
                gray = np.uint8(gray)
            elif (max_out <= 65535):
                gray = np.uint16(gray)
            return gray

        #  如果是多波段
        if (len(image.shape) == 3):
            image_stretch = []
            # print(image.shape[0])
            for i in range(image.shape[0]):
                gray = gray_process(image[i])
                image_stretch.append(gray)
            image_stretch = np.array(image_stretch)
        #  如果是单波段
        else:
            image_stretch = gray_process(image)
        return image_stretch
    def forward(self, lr_ms, b_ms, x_pan):
        '''
        原文蓝线部分
        '''
        B,C,h,w=lr_ms.size()
        hp_pan_4 = x_pan - F.interpolate(F.interpolate(x_pan, scale_factor=1 / 4, mode='bicubic'), scale_factor=4,
                                         mode='bicubic')
        lr_pan = F.interpolate(x_pan, scale_factor=1 / 2, mode='bicubic')
        hp_pan_2 = lr_pan - F.interpolate(F.interpolate(lr_pan, scale_factor=1 / 2, mode='bicubic'), scale_factor=2,
                                          mode='bicubic')
        b_ms_2 = F.interpolate(lr_ms, scale_factor=2, mode='bicubic')

        m1 = self.Memory_2()
        m1 = m1.repeat(1, 1, h, w)
        if self.training == False:
            m1=m1[0].unsqueeze(0)

        s0 = self.head(lr_ms)

        s1 = self.res_block_s1(s0)#+ hp_pan_2
        d1 = s1.detach() * m1

        # d1 = torch.sum(d1,dim=1,keepdim=True) #[B,1,h*2,w*2]
        d1_ = self.cattention_2(d1)
        dd1_ =self.Unet_hp2(d1_)

        s2 = self.res_block_s2(s1)+b_ms_2 +dd1_

        m2= self.Memory_4()
        m2 = m2.repeat(1, 1, h, w)
        if self.training == False:
            m2=m2[0].unsqueeze(0)

        s3 = self.res_block_s3(s2)
        d2 = s3.detach() * m2
        # d2 = torch.sum(d2,dim=1,keepdim=True)
        d2_ =self.cattention_4(d2)
        dd2_ =self.Unet_hp4(d2_)

        s4 = self.res_block_s4(s3) +b_ms +dd2_ # +hp_pan_2

        if self.training:
            loss_fn = nn.KLDivLoss()
            hp_loss=loss_fn(F.log_softmax(dd1_,dim=2),F.softmax(hp_pan_2,dim=2)) + loss_fn(F.log_softmax(dd2_,dim=2),F.softmax(hp_pan_4,dim=2))

            return s4 ,s2, hp_loss
        else:
            # m1_ = m1[:,:,:2,:2]
            # m2_ = m2[:,:,:4,:4]
            # padm1 = F.pad(m1_, (1, 1, 1, 1), "constant", 1)
            # padm2 = F.pad(m2_, (1, 1, 1, 1), "constant", 1)
            # padm1 = padm1.reshape(1, 1, -1, 2 + 2)
            # padm2 = padm2.reshape(1,1,-1,4+2)
            # padm1 = 255 * (padm1 - padm1.min())/(padm1.max() - padm1.min())
            # padm2 = 255 * (padm2 - padm2.min())/(padm2.max() - padm2.min())
            # # padm1 = 255 * (padm1 +1 )/2
            # # padm2 = 255 * (padm2 +1)/2
            # # import pdb
            # # pdb.set_trace()
            # # d = 255 * (d + 1) / 2
            # padm1 = padm1[0].clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy()
            # padm2 = padm2[0].clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy()
            #
            # import skimage.io as io
            # io.imsave(os.path.join('./', "d2_.png"), ((d2_[0]-d2_.min())/(d2_.max()-d2_.min())*255).clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy())
            # io.imsave(os.path.join('./', "dd2_.png"), ((dd2_[0]-dd2_.min())/(dd2_.max()-dd2_.min())*255).clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy())

            return s4,d2_,dd2_



    # def forward(self, x):
    #     '''
    #       x: (b, c, h, w)
    #       embed: (k, t)
    #       t=h*w
    #     '''
    #     b, c, h, w = x.size()
    #     assert c == self.c
    #     k, c = self.k, self.c
    #
    #     x = x.permute(0, 2, 3, 1)
    #     query=x.sum(3)
    #     query = query.reshape(b,-1)  # (b, c)
    #
    #     m = self.units.weight.data  # (k, c)
    #
    #     xn = F.normalize(query, dim=1)  # (b, c)
    #     mn = F.normalize(m, dim=1)  # (k, c)
    #     score = torch.matmul(xn, mn.t())  # (b, k)
    #
    #     soft_label = F.softmax(score, dim=1)
    #     out = torch.matmul(soft_label, m)  # (b, c)
    #     out = out.view(b, h, w, 1).permute(0, 3, 1, 2)
    #
    #     return out, score

    # class Memory_Block(nn.Module):
#     def __init__(self, hdim, kdim, moving_average_rate=0.99):
#         super().__init__()
#
#         self.c = hdim
#         self.k = kdim
#
#         self.moving_average_rate = moving_average_rate
#
#         self.units = nn.Embedding(kdim, hdim)
#
#     def update(self, x, score, m=None):
#         '''
#             x: (n, c)
#             e: (k, c)
#             score: (n, k)
#         '''
#         if m is None:
#             m = self.units.weight.data
#         x = x.detach()
#         embed_ind = torch.max(score, dim=1)[1]  # (n, )
#         embed_onehot = F.one_hot(embed_ind, self.k).type(x.dtype)  # (n, k)
#         embed_onehot_sum = embed_onehot.sum(0) #(1,k)
#         embed_sum = x.transpose(0, 1) @ embed_onehot  # (c, k)
#         embed_mean = embed_sum / (embed_onehot_sum + 1e-6)
#         new_data = m * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)
#         if self.training:
#             self.units.weight.data = new_data
#         return new_data
#
#     def forward(self, x, update_flag=True):
#         '''
#           x: (b, c, h, w)
#           embed: (k, c)
#         '''
#
#         b, c, h, w = x.size()
#         assert c == self.c
#         k, c = self.k, self.c
#
#         x = x.permute(0, 2, 3, 1)
#         x = x.reshape(-1, c)  # (n, c)
#
#         m = self.units.weight.data  # (k, c)
#
#         xn = F.normalize(x, dim=1)  # (n, c)
#         mn = F.normalize(m, dim=1)  # (k, c)
#         score = torch.matmul(xn, mn.t())  # (n, k)
#
#         if update_flag:
#             m = self.update(x, score, m)
#             mn = F.normalize(m, dim=1)  # (k, c)
#             score = torch.matmul(xn, mn.t())  # (n, k)
#
#         soft_label = F.softmax(score, dim=1)
#         out = torch.matmul(soft_label, m)  # (n, c)
#         out = out.view(b, h, w, c).permute(0, 3, 1, 2)
#
#         return out, score