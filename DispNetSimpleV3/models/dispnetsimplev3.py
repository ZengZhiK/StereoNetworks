# -*- coding: utf-8 -*-
"""
author:     tianhe
time  :     2022-09-08 14:40
"""
import torch
from torch import nn
import torch.nn.functional as F


class DispNetSimpleV3(nn.Module):
    def __init__(self, batch_norm=False):
        super().__init__()
        # the extraction part
        self.conv1 = self.conv2d_leakyrelu(6, 64, 7, 2, 3, batch_norm=batch_norm)  # 1/2
        self.conv2 = self.conv2d_leakyrelu(64, 128, 5, 2, 2, batch_norm=batch_norm)  # 1/4
        self.conv3a = self.conv2d_leakyrelu(128, 256, 5, 2, 2, batch_norm=batch_norm)  # 1/8
        self.conv3b = self.conv2d_leakyrelu(256, 256, 3, 1, 1, batch_norm=batch_norm)
        self.conv4a = self.conv2d_leakyrelu(256, 512, 3, 2, 1, batch_norm=batch_norm)  # 1/16
        self.conv4b = self.conv2d_leakyrelu(512, 512, 3, 1, 1, batch_norm=batch_norm)
        self.conv5a = self.conv2d_leakyrelu(512, 512, 3, 2, 1, batch_norm=batch_norm)  # 1/32
        self.conv5b = self.conv2d_leakyrelu(512, 512, 3, 1, 1, batch_norm=batch_norm)
        self.conv6a = self.conv2d_leakyrelu(512, 1024, 3, 2, 1, batch_norm=batch_norm)  # 1/64
        self.conv6b = self.conv2d_leakyrelu(1024, 1024, 3, 1, 1, batch_norm=batch_norm)

        # the expanding part
        self.upconv5 = self.convTranspose2d_leakyrelu(1024, 512, 4, 2, 1)  # conv6b
        self.upconv4 = self.convTranspose2d_leakyrelu(512, 256, 4, 2, 1)  # iconv5
        self.upconv3 = self.convTranspose2d_leakyrelu(256, 128, 4, 2, 1)  # iconv4
        self.upconv2 = self.convTranspose2d_leakyrelu(128, 64, 4, 2, 1)  # iconv3
        self.upconv1 = self.convTranspose2d_leakyrelu(64, 32, 4, 2, 1)  # iconv2

        self.iconv5 = nn.Conv2d(512 + 1 + 512, 512, 3, 1, 1)  # upconv5+pre6+conv5b
        self.iconv4 = nn.Conv2d(256 + 1 + 512, 256, 3, 1, 1)  # upconv4+pre5+conv4b
        self.iconv3 = nn.Conv2d(128 + 1 + 256, 128, 3, 1, 1)  # upconv3+pre4+conv3b
        self.iconv2 = nn.Conv2d(64 + 1 + 128, 64, 3, 1, 1)  # upconv2+pre3+conv2
        self.iconv1 = nn.Conv2d(32 + 1 + 64, 32, 3, 1, 1)  # upconv1+pre2+conv1

        # the predict part
        self.upscale2 = nn.ConvTranspose2d(1, 1, 4, 2, 1)
        self.upscale3 = nn.ConvTranspose2d(1, 1, 4, 2, 1)
        self.upscale4 = nn.ConvTranspose2d(1, 1, 4, 2, 1)
        self.upscale5 = nn.ConvTranspose2d(1, 1, 4, 2, 1)
        self.upscale6 = nn.ConvTranspose2d(1, 1, 4, 2, 1)
        self.pr6 = nn.Conv2d(1024, 1, 3, 1, 1)  # conv6b
        self.pr5 = nn.Conv2d(512, 1, 3, 1, 1)  # iconv5
        self.pr4 = nn.Conv2d(256, 1, 3, 1, 1)  # iconv4
        self.pr3 = nn.Conv2d(128, 1, 3, 1, 1)  # iconv3
        self.pr2 = nn.Conv2d(64, 1, 3, 1, 1)  # iconv2
        self.pr1 = nn.Conv2d(32, 1, 3, 1, 1)  # iconv1

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, imgL, imgR):
        # the extraction part
        conv1 = self.conv1(torch.cat((imgL, imgR), dim=1))
        conv2 = self.conv2(conv1)
        conv3b = self.conv3b(self.conv3a(conv2))
        conv4b = self.conv4b(self.conv4a(conv3b))
        conv5b = self.conv5b(self.conv5a(conv4b))
        conv6b = self.conv6b(self.conv6a(conv5b))

        # the predict part 直接卷积得到视差
        pr6 = self.pr6(conv6b)  # 1/64 [B 1 H/64 W/64]
        iconv5 = self.iconv5(torch.cat((self.upconv5(conv6b), self.upscale6(pr6), conv5b), dim=1))
        pr5 = self.pr5(iconv5)  # 1/32
        iconv4 = self.iconv4(torch.cat((self.upconv4(iconv5), self.upscale5(pr5), conv4b), dim=1))
        pr4 = self.pr4(iconv4)  # 1/16
        iconv3 = self.iconv3(torch.cat((self.upconv3(iconv4), self.upscale4(pr4), conv3b), dim=1))
        pr3 = self.pr3(iconv3)  # 1/8
        iconv2 = self.iconv2(torch.cat((self.upconv2(iconv3), self.upscale3(pr3), conv2), dim=1))
        pr2 = self.pr2(iconv2)  # 1/4
        iconv1 = self.iconv1(torch.cat((self.upconv1(iconv2), self.upscale2(pr2), conv1), dim=1))
        pr1 = self.pr1(iconv1)  # 1/2

        if self.training:
            return pr1, pr2, pr3, pr4, pr5, pr6  # [B 1 H W]
        else:
            pr1 = F.interpolate(pr1, (imgL.shape[2], imgL.shape[3]), mode='bilinear', align_corners=True)
            return pr1  # [B 1 H W]

    def conv2d_leakyrelu(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=False):
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )

    def convTranspose2d_leakyrelu(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
