import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as numpy
from dataloader import FlyingThingsLoader
from model import *
from utils import dataset_loader


lt, rt, ld, lte, rte, ldte = dataset_loader.load_data()
print(lt[0], rt[0], ld[0], lte[0], rte[0], ldte[0])