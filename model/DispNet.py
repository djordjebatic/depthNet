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
import cv2 as cv  # OpenCV

# numeric and plotting libraries
import numpy as np
import matplotlib.pyplot as plt

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

class DispNetSimple(nn.Module):
    ''' Simple DispNet, input is stacked.
    '''
    def __init__(self):
        pass

    def forward(self, left, right):
        pass

class DispNetCorr(nn.Module):
    ''' DispNet with Siamese feature extractors on left and right image
    + horizontal correlation to join them.
    '''
    def __init__(self):
        pass

    def forward(self, left, right):
        pass