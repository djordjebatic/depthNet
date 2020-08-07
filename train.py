import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as numpy
from dataloader import FlyingThingsLoader as FLY
from model.basic import DispNetSimple
from utils import dataset_loader
from torch.autograd import Variable
#from torch.utils.tensorboard import SummaryWriter
import time
from model.loss import multiscaleloss, AverageMeter


DEVICE = torch.device("cuda") # Boilerplate code for using CUDA for faster training
MAX_SUMMARY_IMAGES = 4
LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 64 
NUM_WORKERS = 8
LOSS_WEIGHTS = [[0.32, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005]]

assert MAX_SUMMARY_IMAGES <= BATCH_SIZE

torch.manual_seed(1)
torch.cuda.manual_seed(1)

def train_sample():
    # Initialize the Tensorboard summary. Logs will end up in runs directory
    #summary_writer = SummaryWriter()

    losses = AverageMeter()

    left_imgs_train, right_imgs_train, left_disps_train, left_imgs_test, right_imgs_test, left_disps_test = dataset_loader.load_data()
    print('loaded data')

    train_loader = torch.utils.data.DataLoader(
        FLY.FlyingThingsDataloader(left_imgs_train[:100], right_imgs_train[:100], left_disps_train[:100], True),
        batch_size=64, shuffle=False, num_workers=8, drop_last=False
    )

    test_loader = torch.utils.data.DataLoader(
        FLY.FlyingThingsDataloader(left_imgs_test[:25], right_imgs_test[:25], left_disps_test[:25], False),
        batch_size=64, shuffle=False, num_workers=4, drop_last=False
    )

    model = DispNetSimple().cuda()
    #model = nn.DataParallel(model).cuda()
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    optimizer = optim.Adam(model.parameters(), lr=LR)
    total_train_loss = 0

    model.train()
    start = time.time()
    
    print('Training started')
    
    for batch_idx, (imgL, imgR, dispL) in enumerate(train_loader):
        print('Batch ', batch_idx)

        criterion = multiscaleloss(7, 1, LOSS_WEIGHTS[batch_idx], loss='L1', sparse=False)

        imgL = Variable(torch.FloatTensor(imgL).cuda(), requires_grad=False)
        imgR = Variable(torch.FloatTensor(imgR).cuda(), requires_grad=False)
        dispL = Variable(torch.FloatTensor(dispL).cuda(), requires_grad=False)
        input_cat = Variable(torch.cat((imgL, imgR), 1), requires_grad=False)
        # maximum disparity
        mask = dispL < 192
        mask.detach_()

        optimizer.zero_grad()
        output = model(input_cat)

        print(dispL[0].shape)
        print(output[0].shape)
        print(output[1].shape)
        print(output[2].shape)
        print(output[3].shape)

        loss = criterion(output, dispL)
        if type(loss) is list or type(loss) is tuple:
            loss = torch.sum(loss)

        losses.update(loss.data.item(), dispL.size(0))

        loss.backward()
        optimizer.step()

        print('Loss {loss.val:.3f} ({loss.avg:.3f})', loss=losses)
    end = time.time()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    train_sample()