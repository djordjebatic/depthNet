import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as numpy
from dataloader import FlyingThingsLoader as FLY
from dataloader.KITTILoader import KITTILoader
from model.basic import DispNetSimple
from model.DispNetV2 import DispNetV2
#from model.loss import multiscale_loss
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from utils import dataset_loader
import time
import copy


DEVICE = torch.device("cuda")
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
            FLY.FlyingThingsDataloader(left_imgs_val[:1000], right_imgs_val[:1000], left_disps_val[:1000], False),
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


def model_init(dual_gpu=False, v2=False):
    if v2:
        model = DispNetV2().to(DEVICE)
    else:
        model = DispNetSimple().to(DEVICE)
    if dual_gpu:
        model = nn.DataParallel(model)
    print('Model initialized.')
    print('Number of model parameters:\t{}'.format(sum([p.data.nelement() for p in model.parameters()])))
    return model


def calculate_loss(output, disp_true, weights = None, mask=None):

    loss = 0
    pr1, pr2, pr3, pr4, pr5, pr6 = output
    # predict flow upsampling
    out_size = pr1.shape[-2:]
    out1 = pr1
    out2 = F.interpolate(pr2, out_size, mode='bilinear')
    out3 = F.interpolate(pr3, out_size, mode='bilinear')
    out4 = F.interpolate(pr4, out_size, mode='bilinear')
    out5 = F.interpolate(pr5, out_size, mode='bilinear')
    out6 = F.interpolate(pr6, out_size, mode='bilinear')

    #kitti mode
    if mask != None:
        # apply masks
        out1 = torch.squeeze(out1, 1)[mask]
        out2 = torch.squeeze(out2, 1)[mask]
        out3 = torch.squeeze(out3, 1)[mask]
        out4 = torch.squeeze(out4, 1)[mask]
        out5 = torch.squeeze(out5, 1)[mask]
        out6 = torch.squeeze(out6, 1)[mask]

        disp_true = disp_true[mask]

    else:
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
    for w, o in zip(weights, outs):
        loss_delta = w * F.smooth_l1_loss(o, disp_true, size_average=True)
        loss += loss_delta


    return loss #* 0.0002 * (out1 != 0).sum()

def main(model, train_loader, val_loader):


    start = time.time()

    writer = SummaryWriter()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    print('Training loop started.')


    for epoch in range(1, EPOCHS):
        print("Epoch {}/{}\n----------".format(epoch, EPOCHS))
        for phase in ['train', 'val']:
            min_loss = 10000
            total_loss = 0

            if phase == 'train':
                model.train()  # Set model to training mode
                loader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                loader = val_loader

            for batch_idx, (imgL, imgR, dispL) in enumerate(loader):

                imgL = Variable(torch.FloatTensor(imgL).to(DEVICE))
                imgR = Variable(torch.FloatTensor(imgR).to(DEVICE))
                disp_true = Variable(torch.FloatTensor(dispL).to(DEVICE))
                input_cat = Variable(torch.cat((imgL, imgR), 1))
                
                #mask = (disp_true > 0)
                #mask.detach_()
                #mask = None

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(input_cat)

                    if phase == 'train':
                        loss = calculate_loss(output, disp_true, mask=mask)
                        loss.backward()
                        optimizer.step()     

                    else:
                        disp = torch.squeeze(output, 1)
                        loss = F.smooth_l1_loss(disp_true, disp)
     

                total_loss += loss

            if phase=='val' and total_loss < min_loss: 
                min_loss = total_loss
                torch.save(model.state_dict(), MODEL_PTH + str(epoch) + '_dispnet.pth')

            writer.add_scalar('loss/' + phase, loss, epoch)
            print('{} epoch {} loss: {:.3f}'.format(phase, epoch, loss))

        print('\n')
    print('time elapsed: {:.3f}'.format(time.time() - start))
    #print('Total loss is: {:.3f}'.format(total_train_loss/EPOCHS))
    del model, imgL, imgR, disp_true

    writer.close()


def validation_simple(model, val_loader):
    # TODO include validation inside the training loop for progress tracking
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

    #root = 'SAMPLE_BATCH'
    train_loader, val_loader = make_data_loaders('driving')

    model = model_init(dual_gpu=True, v2=True)
    #state_dict_filename = "saved_models/46_dispnet.pth"
    #state_dict = torch.load(state_dict_filename)
    #model.load_state_dict(state_dict)
    print("Model loaded.")

    #model = model_init()
    main(model, train_loader, val_loader)
    #validation_simple(model, val_loader)
