import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as numpy
from dataloader.StereoDatasetLoader import get_data_loaders
from model.basic import DispNetSimple
from model.loss import multiscale_loss
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import time
from PIL import Image
import numpy as np


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MAX_SUMMARY_IMAGES = 4
LR = 3e-4
EPOCHS = 300
BATCH_SIZE = 16
NUM_WORKERS = 8
LOSS_WEIGHTS = [[0.32, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005]]
MODEL_PTH = 'saved_models/'

assert MAX_SUMMARY_IMAGES <= BATCH_SIZE

torch.manual_seed(1)
torch.cuda.manual_seed(1)

def model_init(dual_gpu=False):
    model = DispNetSimple().to(DEVICE)
    if dual_gpu:
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print('Model initialized.')
    print('Number of model parameters:\t{}'.format(sum([p.data.nelement() for p in model.parameters()])))
    return model, optimizer


def train_sample(model, optimizer, train_loader, root = 'FlyingThings3D_subset'):

    total_train_loss = 0

    model.train()
    start = time.time()

    writer = SummaryWriter()

    print('Training loop started.')

    for epoch in range(1, EPOCHS):
        for batch_idx, (imgL, imgR, dispL) in enumerate(train_loader):

            imgL = Variable(torch.FloatTensor(imgL).to(DEVICE), requires_grad=False)
            imgR = Variable(torch.FloatTensor(imgR).to(DEVICE), requires_grad=False)
            disp_true = Variable(torch.FloatTensor(dispL).to(DEVICE), requires_grad=False)
            input_cat = Variable(torch.cat((imgL, imgR), 1), requires_grad=False)

            optimizer.zero_grad()
            output = model(input_cat)

            loss = multiscale_loss(output, disp_true)

            total_train_loss += loss

            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), MODEL_PTH + str(epoch) + '_dispnet.pth')
        total_train_loss += loss
        writer.add_scalar('loss/train', loss, epoch)
        print('Epoch {} loss: {:.3f} Time elapsed: {:.3f}'.format(epoch, loss, time.time() - start))

    print('Total loss is: {:.3f}'.format(total_train_loss/EPOCHS))
    del model, imgL, imgR, disp_true
    print('Total time elapsed: {:.3f}'.format(time.time() - start))

    writer.close()


def validation_simple(model, val_loader):
    model.eval()

    total_validation_loss = 0
    start = time.time()
    for batch_idx, (imgL, imgR, dispL) in enumerate(val_loader):
        imgL, imgR, disp_true = imgL.to(DEVICE), imgR.to(DEVICE), dispL.to(DEVICE)

        with torch.no_grad():
            disp = model(torch.cat((imgL, imgR), 1))
            disp = torch.squeeze(disp)

        #print(disp.shape, disp_true.shape)
        #disp = F.upsample(disp, size=(BATCH_SIZE, 512, 960), mode='bilinear')
        loss = F.smooth_l1_loss(disp_true, disp)
        total_validation_loss += loss
        print('Batch {} val loss: {:.3f} Time elapsed: {:.3f}'.format(batch_idx, loss, time.time() - start))

    print('\nTotal val loss: {:.3f} Time elapsed: {:.3f}'.format(total_validation_loss/len(val_loader), time.time() - start))

    return loss.data.cpu()


class WrappedModel(nn.Module):
    def __init__(self):
        super(WrappedModel, self).__init__()
        self.module = DispNetSimple().to(DEVICE)
    def forward(self, x):
        return self.module(x)


if __name__ == '__main__':
    torch.cuda.empty_cache()

    root = 'SAMPLE_BATCH'
    train_loader, val_loader = get_data_loaders(BATCH_SIZE, NUM_WORKERS)

    # dual_gpu = torch.cuda.device_count() > 1
    # model, optimizer = model_init(dual_gpu)
    # train_sample(model, optimizer, train_loader)

    state_dict_filename = "saved_models/46_dispnet.pth"
    state_dict = torch.load(state_dict_filename)
    model = WrappedModel()
    model.load_state_dict(state_dict)
    print("Model loaded.")

    validation_simple(model, val_loader)
