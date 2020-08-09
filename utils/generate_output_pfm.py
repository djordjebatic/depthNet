import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from model.basic import DispNetSimple
from utils.python_pfm import *
import time
from PIL import Image
import matplotlib.pyplot as plt

data_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class WrappedModel(nn.Module):
    def __init__(self):
        super(WrappedModel, self).__init__()
        self.module = DispNetSimple().to(device) # that I actually define.
    def forward(self, x):
        return self.module(x)


def load_model(model_path):
    print("Loading model")
    model = WrappedModel().to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    print("Model loaded")
    return model


def infer(model, imgL, imgR):
    model.eval()

    imgL = torch.FloatTensor(imgL).to(device)
    imgR = torch.FloatTensor(imgR).to(device)

    with torch.no_grad():
        input_cat = torch.cat((imgL, imgR), 1)
        disp = model(input_cat)

    pred_disp = torch.nn.functional.interpolate(disp, (540, 960), mode='bilinear').cpu().numpy()
    pred_disp = np.squeeze(pred_disp)
    return pred_disp


def generate_output(epoch, img_idx):

    model_path = "saved_models/" + epoch + "_dispnet.pth"
    model = load_model(model_path)

    left_img_path = "FlyingThings3D_subset/val/image_clean/left/" + img_idx + ".png"
    right_img_path = "FlyingThings3D_subset/val/image_clean/right/" + img_idx + ".png"
    th, tw = 512, 960
    imgL = Image.open(left_img_path).convert('RGB').resize((tw, th))
    imgR = Image.open(right_img_path).convert('RGB').resize((tw, th))
    imgL = data_transforms(imgL).unsqueeze(0)
    imgR = data_transforms(imgR).unsqueeze(0)

    start_time = time.time()
    pred_disp = infer(model, imgL,imgR)

    print(pred_disp.shape)

    writePFM(img_idx + ".pfm", pred_disp)

    print('time = %.2f' %(time.time() - start_time))

    f = plt.figure()
    f.add_subplot(2,1,1)
    plt.imshow(np.asarray(imgL))
    f.add_subplot(2,1,2)
    plt.imshow(pred_disp)
    plt.show()
    print('total time = %.2f' %(time.time() - start_time))

epoch = "46"
img_idx = "0000799"
generate_output(epoch, img_idx)
pred_disp, _ = readPFM(img_idx + ".pfm")

print('min: %f, max: %f, mean: %f' % (np.min(pred_disp), np.max(pred_disp), np.mean(pred_disp)))
plt.imshow(pred_disp)
plt.show()
