# Python's native libraries
import time
import os

# deep learning/vision libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.nn.init import kaiming_normal_, constant_

# numeric and plotting libraries
import numpy as np

'''
    DispNet:

    Name        Kernel  Str.    Ch I/O      InpRes      OutRes      Input
    ---------------------------------------------------------------------------------
    conv1       7×7     2       6/64        768×384     384×192     Images
    conv2       5×5     2       64/128      384×192     192×96      conv1
    conv3a      5×5     2       128/256     192×96      96×48       conv2
    conv3b      3×3     1       256/256     96×48       96×48       conv3a
    conv4a      3×3     2       256/512     96×48       48×24       conv3b
    conv4b      3×3     1       512/512     48×24       48×24       conv4a
    conv5a      3×3     2       512/512     48×24       24×12       conv4b
    conv5b      3×3     1       512/512     24×12       24×12       conv5a
    conv6a      3×3     2       512/1024    24×12       12×6        conv5b
    conv6b      3×3     1       1024/1024   12×6        12×6        conv6a
    pr6+loss6   3×3     1       1024/1      12×6        12×6        conv6b
    upconv5     4×4     2       1024/512    12×6        24×12       conv6b
    iconv5      3×3     1       1025/512    24×12       24×12       upconv5+pr6+conv5b
    pr5+loss5   3×3     1       512/1       24×12       24×12       iconv5
    upconv4     4×4     2       512/256     24×12       48×24       iconv5
    iconv4      3×3     1       769/256     48×24       48×24       upconv4+pr5+conv4b
    pr4+loss4   3×3     1       256/1       48×24       48×24       iconv4
    upconv3     4×4     2       256/128     48×24       96×48       iconv4
    iconv3      3×3     1       385/128     96×48       96×48       upconv3+pr4+conv3b
    pr3+loss3   3×3     1       128/1       96×48       96×48       iconv3
    upconv2     4×4     2       128/64      96×48       192×96      iconv3
    iconv2      3×3     1       193/64      192×96      192×96      upconv2+pr3+conv2
    pr2+loss2   3×3     1       64/1        192×96      192×96      iconv2
    upconv1     4×4     2       64/32       192×96      384×192     iconv2
    iconv1      3×3     1       97/32       384×192     384×192     upconv1+pr2+conv1
    pr1+loss1   3×3     1       32/1        384×192     384×192     iconv1
'''


def conv_block(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size-1)//2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True)
    )


def deconv(in_channels, out_channels, kernel_size=4):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels,
                           kernel_size, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1, inplace=True)
    )


def predict_disp(in_channels):
    return nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1, bias=False)


class DispNetSimple(nn.Module):
    ''' Simple DispNet, input is stacked.
    '''

    def __init__(self):
        super(DispNetSimple, self).__init__()

        self.conv1 = conv_block(6, 64, kernel_size=7, stride=2)
        self.conv2 = conv_block(64, 128, kernel_size=5, stride=2)
        self.conv3a = conv_block(128, 256, kernel_size=5, stride=2)
        self.conv3b = conv_block(256, 256)
        self.conv4a = conv_block(256, 512, stride=2)
        self.conv4b = conv_block(512, 512)
        self.conv5a = conv_block(512, 512, stride=2)
        self.conv5b = conv_block(512, 512)
        self.conv6a = conv_block(512, 1024, stride=2)
        self.conv6b = conv_block(1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(512, 256)
        self.deconv3 = deconv(256,  128)
        self.deconv2 = deconv(128,  64)
        self.deconv1 = deconv(64,  32)

        self.predict_disp6 = predict_disp(1024)
        self.predict_disp5 = predict_disp(512)
        self.predict_disp4 = predict_disp(256)
        self.predict_disp3 = predict_disp(128)
        self.predict_disp2 = predict_disp(64)
        self.predict_disp1 = predict_disp(32)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)

        self.iconv5 = nn.Conv2d(1025, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.iconv4 = nn.Conv2d(769, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.iconv3 = nn.Conv2d(385, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.iconv2 = nn.Conv2d(193, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.iconv1 = nn.Conv2d(97, 32, kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3b = self.conv3b(self.conv3a(out_conv2))
        out_conv4b = self.conv4b(self.conv4a(out_conv3b))
        out_conv5b = self.conv5b(self.conv5a(out_conv4b))
        out_conv6b = self.conv6b(self.conv6a(out_conv5b))

        pr6 = self.predict_disp6(out_conv6b)
        pr6_up = self.upsampled_flow6_to_5(pr6)

        deconv5 = self.deconv5(out_conv6b)
        iconv5 = self.iconv5(torch.cat([deconv5, pr6_up, out_conv5b], dim=1))
        pr5 = self.predict_disp5(iconv5)
        pr5_up = self.upsampled_flow5_to_4(pr5)

        deconv4 = self.deconv4(iconv5)
        iconv4 = self.iconv4(torch.cat([deconv4, pr5_up, out_conv4b], dim=1))
        pr4 = self.predict_disp4(iconv4)
        pr4_up = self.upsampled_flow4_to_3(pr4)

        deconv3 = self.deconv3(iconv4)
        iconv3 = self.iconv3(torch.cat([deconv3, pr4_up, out_conv3b], dim=1))
        pr3 = self.predict_disp3(iconv3)
        pr3_up = self.upsampled_flow3_to_2(pr3)

        deconv2 = self.deconv2(iconv3)
        iconv2 = self.iconv2(torch.cat([deconv2, pr3_up, out_conv2], dim=1))
        pr2 = self.predict_disp2(iconv2)
        pr2_up = self.upsampled_flow2_to_1(pr2)

        deconv1 = self.deconv1(iconv2)
        iconv1 = self.iconv1(torch.cat([deconv1, pr2_up, out_conv1], dim=1))
        pr1 = self.predict_disp1(iconv1)


        if self.training:
            return pr1, pr2, pr3, pr4, pr5, pr6
        else:
            return pr1
