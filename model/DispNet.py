''' Module defines Disparity Networks.
'''

# Python's native libraries
import time
import os

# deep learning/vision libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

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

def conv_block(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding_mode="same", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
    )

def predict_flow(in_planes):
    # return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)  # NVIDIA FlowNetS
    return nn.Conv2d(in_planes,1,kernel_size=3,stride=1,padding=1,bias=True)    # DispNetS

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )


class DispNetSimple(nn.Module):
    ''' Simple DispNet, input is stacked.
    '''
    def __init__(self):
        super().__init__()

        self.conv1 = conv_block(6, 64, 7, 2)
        self.conv2 = conv_block(64, 128, 5, 2)
        self.conv3a = conv_block(128, 256, 5, 2)
        self.conv3b = conv_block(256, 256, 3, 1)
        self.conv4a = conv_block(256, 512, 3, 2)
        self.conv4b = conv_block(512, 512, 3, 1)
        self.conv5a = conv_block(512, 512, 3, 2)
        self.conv5b = conv_block(512, 512, 3, 1)
        self.conv6a = conv_block(512, 1024, 3, 2)
        self.conv6b = conv_block(1024, 1024, 3, 1)
        self.pr6 = predict_flow(1024)
        self.pr5 = predict_flow(512)
        self.pr4 = predict_flow(256)
        self.pr3 = predict_flow(128)
        self.pr2 = predict_flow(64)
        self.pr1 = predict_flow(32)
        def deconv(in_planes, out_planes):

        self.upconv5 = deconv(1024, 512)
        self.upconv4 = deconv(512, 256)
        self.upconv3 = deconv(256, 128)
        self.upconv2 = deconv(128, 64)
        self.upconv1 = deconv(64, 32)

        # TODO iconv?

        # TODO try bilinear upsample

    def forward(self, features):
        # left, right = features

        pass

class DispNet(nn.Module):

    def __init__(self):
        pass

    def forward(self, left, right):
        pass