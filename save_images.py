from __future__ import print_function
from __future__ import division
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import models
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import time
from tensorboardX import SummaryWriter
from glob import glob
from util import *
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

path = '../OutDistribution/'
kwargs = {'num_workers': 3, 'pin_memory': True}

simple_transform = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.Grayscale(num_output_channels=3),
                                       transforms.ToTensor()])
out = ImageFolder(path,simple_transform)
out_data_gen = torch.utils.data.DataLoader(out,shuffle=True,batch_size=128,num_workers=kwargs['num_workers'])
dataset_sizes = {'out':len(out_data_gen.dataset)}
dataloaders = {'out':out_data_gen}

count = 0
for data in dataloaders['valid']:
    count += 1
    # get the inputs
    inputs, _ = data

    # wrap them in Variable
    if torch.cuda.is_available():
        inputs = Variable(inputs.cuda())
    else:
        inputs = Variable(inputs)
    
    if count == 1:
        torchvision.utils.save_image(inputs.data, './imgs/POSTER_OUT_IMGS.jpg'.format(epoch), nrow=8, padding=2)
