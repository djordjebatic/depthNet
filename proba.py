import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from model.basic import DispNetSimple
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from utils.python_pfm import *

data_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

device = torch.device("cuda:1")

class WrappedModel(nn.Module):
    def __init__(self):
        super(WrappedModel, self).__init__()
        self.module = DispNetSimple().to(device) # that I actually define.
    def forward(self, x):
        return self.module(x)

def test(model, imgL, imgR):
    model.eval()

    imgL = torch.FloatTensor(imgL).to(device)
    imgR = torch.FloatTensor(imgR).to(device)   

    with torch.no_grad():
        input_cat = torch.cat((imgL, imgR), 1)
        #print(input_cat.shape)
        disp = model(input_cat)
        #disp = torch.squeeze(disp)

    pred_disp = torch.nn.functional.interpolate(disp, (540, 960), mode='bilinear').cpu().numpy()
    pred_disp = np.squeeze(pred_disp)
    return pred_disp
 
def generate_output(epoch, img_path):

    print("Loading model")
    # then I load the weights I save from previous code:
    model = WrappedModel().to(device)
    state_dict = torch.load("saved_models/" + epoch + "_dispnet.pth")
    model.load_state_dict(state_dict)

    print("Model loaded")

    left_img = "FlyingThings3D_subset/val/image_clean/left/" + img_path + ".png"
    right_img = "FlyingThings3D_subset/val/image_clean/right/" + img_path + ".png"
    
    th, tw = 512, 960
    imgL_o = Image.open(left_img).convert('RGB').resize((tw, th))
    imgR_o = Image.open(right_img).convert('RGB').resize((tw, th))

    imgL = data_transforms(imgL_o).unsqueeze(0)
    imgR = data_transforms(imgR_o).unsqueeze(0)


    start_time = time.time()
    pred_disp = test(model, imgL,imgR)
    
    print(pred_disp.shape)

    #resize 540, 960
    writePFM(img_path + ".pfm", pred_disp)

    print('time = %.2f' %(time.time() - start_time))

    f = plt.figure()
    f.add_subplot(2,1,1)
    plt.imshow(np.asarray(imgL_o))
    f.add_subplot(2,1,2)
    plt.imshow(pred_disp)
    plt.show()
    print('total time = %.2f' %(time.time() - start_time))

generate_output("46", "0000799")
pred_disp, _ = readPFM("0000799.pfm")

print('min: %f, max: %f, mean: %f' % (np.min(pred_disp), np.max(pred_disp), np.mean(pred_disp)))
plt.imshow(pred_disp)
plt.show()
