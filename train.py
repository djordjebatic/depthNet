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
from PIL import Image
import numpy as np


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MAX_SUMMARY_IMAGES = 4
LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 64
NUM_WORKERS = 8
LOSS_WEIGHTS = [[0.32, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005]]
MODEL_PTH = 'saved_models/'

assert MAX_SUMMARY_IMAGES <= BATCH_SIZE

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def make_data_loaders(root = 'FlyingThings3D_subset'):
    'Loads the train and val datasets'
    left_imgs_train, right_imgs_train, left_disps_train, left_imgs_val, right_imgs_val, left_disps_val = dataset_loader.load_data(root)

    train_loader = torch.utils.data.DataLoader(
        FLY.FlyingThingsDataloader(left_imgs_train[:50], right_imgs_train[:50], left_disps_train[:50], True),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )

    val_loader = torch.utils.data.DataLoader(
        FLY.FlyingThingsDataloader(left_imgs_val[:25], right_imgs_val[:25], left_disps_val[:25], False),
        batch_size=64, shuffle=False, num_workers=4, drop_last=False
    )

    print('Data loaded.')
    return (train_loader, val_loader)


def model_init():
    model = DispNetSimple().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print('Model initialized.')
    print('Number of model parameters:\t{}'.format(sum([p.data.nelement() for p in model.parameters()])))
    return model, optimizer


def calculate_loss(output, disp_true, weights = None):
    pr1, pr2, pr3, pr4, pr5, pr6 = output

    # predict flow upsampling
    out_size = pr1.shape[-2:]
    out1 = pr1
    out2 = F.interpolate(pr2, out_size, mode='bilinear')
    out3 = F.interpolate(pr3, out_size, mode='bilinear')
    out4 = F.interpolate(pr4, out_size, mode='bilinear')
    out5 = F.interpolate(pr5, out_size, mode='bilinear')
    out6 = F.interpolate(pr6, out_size, mode='bilinear')

    # squeeze
    out1 = torch.squeeze(out1, 1)
    out2 = torch.squeeze(out2, 1)
    out3 = torch.squeeze(out3, 1)
    out4 = torch.squeeze(out4, 1)
    out5 = torch.squeeze(out5, 1)
    out6 = torch.squeeze(out6, 1)

    # weights
    if weights is None:
        weights = [0.0025, 0.005, 0.01, 0.02, 0.08, 0.32]

    outs = (out6, out5, out4, out3, out2, out1)
    loss = 0
    for w, o in zip(weights, outs):
        loss_delta = w * F.smooth_l1_loss(o, disp_true, size_average=True)
        loss += loss_delta

    # loss = F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True)

    return loss

def train_sample(model, optimizer, train_loader, val_loader, root = 'FlyingThings3D_subset'):

    total_train_loss = 0

    model.train()
    start = time.time()

    # TODO Initialize the Tensorboard summary. Logs will end up in runs directory
    #summary_writer = SummaryWriter()

    print('Training loop started.')

    for epoch in range(1, EPOCHS):
        for batch_idx, (imgL, imgR, dispL) in enumerate(train_loader):

            #criterion = multiscaleloss(7, 1, LOSS_WEIGHTS[batch_idx], loss='L1', sparse=False)

            imgL = Variable(torch.FloatTensor(imgL).to(DEVICE), requires_grad=False)
            imgR = Variable(torch.FloatTensor(imgR).to(DEVICE), requires_grad=False)
            disp_true = Variable(torch.FloatTensor(dispL).to(DEVICE), requires_grad=False)
            input_cat = Variable(torch.cat((imgL, imgR), 1), requires_grad=False)

            optimizer.zero_grad()
            output = model(input_cat)

            loss = calculate_loss(output, disp_true)

            '''
                loss = criterion(output, dispL)
                if type(loss) is list or type(loss) is tuple:
                    loss = torch.sum(loss)

                losses.update(loss.data.item(), disp_true.size(0))
            '''

            total_train_loss += loss

            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), MODEL_PTH + str(epoch) + '_dispnet.pth')
        total_train_loss += loss
        print('Epoch {} loss: {:.3f} Time elapsed: {:.3f}'.format(epoch, loss, time.time() - start))

    print('Total loss is: {:.3f}'.format(total_train_loss/EPOCHS))
    del model, imgL, imgR, disp_true
    print('Total time elapsed: {:.3f}'.format(time.time() - start))



if __name__ == '__main__':
    torch.cuda.empty_cache()

    root = 'SAMPLE_BATCH'
    train_loader, val_loader = make_data_loaders(root)

    model, optimizer = model_init()

    train_sample(model, optimizer, train_loader, val_loader)