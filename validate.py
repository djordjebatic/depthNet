import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
from dataloader import FlyingThingsLoader as FLY
from dataloader.KITTILoader import KITTILoader
from model.basic import DispNetSimple
#from model.loss import multiscale_loss
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from utils import dataset_loader
from utils.metrics import *

import time
import copy


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MAX_SUMMARY_IMAGES = 4
LR = 3e-4
EPOCHS = 300
BATCH_SIZE = 16
NUM_WORKERS = 8
MODEL_PTH = 'saved_models/encoderv2_'


assert MAX_SUMMARY_IMAGES <= BATCH_SIZE

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def make_data_loaders(root = 'FlyingThings3D_subset'):
    'Loads the train and val datasets'
    left_imgs_train, right_imgs_train, left_disps_train, left_imgs_val, right_imgs_val, left_disps_val = dataset_loader.load_data(root)

    print(len(left_disps_train), len(left_imgs_train))

    if root == 'FlyingThings3D_subset' or root == 'driving':
        train_loader = torch.utils.data.DataLoader(
            FLY.FlyingThingsDataloader(left_imgs_train[:12000], right_imgs_train[:12000], left_disps_train[:12000], True),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
        )

        val_loader = torch.utils.data.DataLoader(
            FLY.FlyingThingsDataloader(left_imgs_val[:1000], right_imgs_val[1000], left_disps_val[1000], False),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
        )

    elif root == 'KITTI':
        train_loader = torch.utils.data.DataLoader(
            KITTILoader(left_imgs_train, right_imgs_train, left_disps_train, True),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
        )

        val_loader = torch.utils.data.DataLoader(
            KITTILoader(left_imgs_val, right_imgs_val, left_disps_val, False),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
        )

    print('Data loaded.')
    return train_loader, val_loader



class WrappedModel(nn.Module):
    def __init__(self):
        super(WrappedModel, self).__init__()
        self.module = DispNetSimple().to(DEVICE)
    def forward(self, x):
        return self.module(x)


def get_metrics(model, loader):
    model.eval()

    rmse = 0
    mrae = 0

    for batch_idx, (imgL, imgR, disp_gt) in enumerate(loader):
        imgL = imgL.to(DEVICE)
        imgR = imgR.to(DEVICE)
        disp_true = dispL.to(DEVICE)
        input_cat = torch.cat((imgL, imgR), 1)
        disp_our = model(input_cat)
        disp_our = torch.squeeze(disp_our)

        rmse += root_mean_square_error(disp_our, disp_gt)
        mrae += relative_absolute_error(disp_our, disp_gt)

    rmse /= len(loader)
    mrae /= len(loader)

    return rmse, mrae

if __name__ == '__main__':
    torch.cuda.empty_cache()

    train_loader, val_loader = make_data_loaders()

    state_dict_filename = "saved_models/46_dispnet.pth"
    state_dict = torch.load(state_dict_filename)

    model = WrappedModel().to(DEVICE)
    model.load_state_dict(state_dict)
    print("Model loaded.")

    rmse, mrae = get_metrics(model, val_loader)
    print("RMSE:", rmse, "\nMRAE:", mrae)
