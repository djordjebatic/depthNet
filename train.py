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

assert MAX_SUMMARY_IMAGES <= BATCH_SIZE

torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Initialize the Tensorboard summary. Logs will end up in runs directory
#summary_writer = SummaryWriter()

#losses = AverageMeter()


def make_data_loaders(root):
    'Loads the train and val datasets'
    left_imgs_train, right_imgs_train, left_disps_train, left_imgs_val, right_imgs_val, left_disps_val = dataset_loader.load_data(root)
    print('loaded data')

    train_loader = torch.utils.data.DataLoader(
        FLY.FlyingThingsDataloader(left_imgs_train[:100], right_imgs_train[:100], left_disps_train[:100], True),
        batch_size=64, shuffle=False, drop_last=False
    )

    val_loader = torch.utils.data.DataLoader(
        FLY.FlyingThingsDataloader(left_imgs_val[:25], right_imgs_val[:25], left_disps_val[:25], False),
        batch_size=64, shuffle=False, num_workers=4, drop_last=False
    )
    return (train_loader, val_loader)


def train_sample(root = 'FlyingThings3D_subset'):
    
    train_loader, val_loader = make_data_loaders(root)

    model = DispNetSimple().to(DEVICE)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    optimizer = optim.Adam(model.parameters(), lr=LR)

    total_train_loss = 0

    model.train()
    start = time.time()
    
    print('Training started')
    
    for epoch in range(1, EPOCHS):
        for batch_idx, (imgL, imgR, dispL) in enumerate(train_loader):

            #criterion = multiscaleloss(7, 1, LOSS_WEIGHTS[batch_idx], loss='L1', sparse=False)

            imgL = Variable(torch.FloatTensor(imgL).to(DEVICE), requires_grad=False)
            imgR = Variable(torch.FloatTensor(imgR).to(DEVICE), requires_grad=False)
            disp_true = Variable(torch.FloatTensor(dispL).to(DEVICE), requires_grad=False)
            input_cat = Variable(torch.cat((imgL, imgR), 1), requires_grad=False)
            # maximum disparity
            mask = disp_true < 192
            mask.detach_()

            optimizer.zero_grad()
            output = model(input_cat)

            loss = calculate_loss(output, mask, disp_true)

            '''
                loss = criterion(output, dispL)
                if type(loss) is list or type(loss) is tuple:
                    loss = torch.sum(loss)

                losses.update(loss.data.item(), disp_true.size(0))
            '''

            total_train_loss += loss
            
            loss.backward()
            optimizer.step()

        print('Epoch {} loss is: {:.3f}'.format(epoch, loss))

    print('Total loss is: {:.3f}'.format(total_train_loss/EPOCHS))
    end = time.time()
    del model, imgL, imgR, disp_true
    print('Time elapsed: {:.3f}'.format(end - start))


def calculate_loss(output, mask, disp_true):

    '''print(disp_true[0].shape)
    print(mask.shape)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)
    print(output[4].shape)
    print(output[5].shape)'''


    output1 = torch.squeeze(output[0], 1)
    output2 = torch.squeeze(output[1], 1)
    output3 = torch.squeeze(output[2], 1)
    ''' output4 = torch.squeeze(output[3], 1)
    output5 = torch.squeeze(output[4], 1)
    output6 = torch.squeeze(output[5], 1)'''

    #print(output1.shape)
    #img = Image.fromarray(np.asarray(output1[0].numpy()), 'RGB')
    #img.save('img.png')
    #img.show()

    loss = F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) 
    #+ 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 

    return loss

if __name__ == '__main__':
    torch.cuda.empty_cache()

    root = 'SAMPLE_BATCH'
    train_sample(root)