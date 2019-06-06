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
BATCH_SIZE=128

simple_transform = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.Grayscale(num_output_channels=3),
                                       transforms.ToTensor()])
cat_transform = transforms.Compose([transforms.Resize((224,224)),
                                       #transforms.Grayscale(num_output_channels=3),  #HACK
                                       transforms.ToTensor()])
flipped_transform = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.Grayscale(num_output_channels=3),  #HACK
                                       transforms.RandomRotation(degrees=90), #HACK
                                       transforms.ToTensor()])
out = ImageFolder(path,simple_transform)
path_cat = '../PetImages/train/'
path_hand = '../Bone/valid/'
out_data_gen = torch.utils.data.DataLoader(out,shuffle=True,batch_size=BATCH_SIZE,num_workers=kwargs['num_workers'])
cat_distr = ImageFolder(path_cat,cat_transform)
cat_distr_data_gen = torch.utils.data.DataLoader(cat_distr,batch_size=BATCH_SIZE,num_workers=kwargs['num_workers'])
flipped_out_distr = ImageFolder(out,flipped_transform)
flipped_out_distr_data_gen = torch.utils.data.DataLoader(flipped_out_distr,batch_size=BATCH_SIZE,num_workers=kwargs['num_workers'])
hand_distr = ImageFolder(path_hand,simple_transform)
hand_distr_data_gen = torch.utils.data.DataLoader(hand_distr,batch_size=BATCH_SIZE,num_workers=kwargs['num_workers'])
dataloaders = {
    'out':out_data_gen,
    'cat':cat_distr_data_gen,
    'hand':hand_distr_data_gen,
    'flipped':flipped_out_distr_data_gen}

"""count = 0
for data in dataloaders['out']:
    count += 1
    # get the inputs
    inputs, _ = data

    # wrap them in Variable
    if torch.cuda.is_available():
        inputs = Variable(inputs.cuda())
    else:
        inputs = Variable(inputs)
    
    if count == 1:
        torchvision.utils.save_image(inputs.data, './imgs/POSTER_OUT_IMGS.jpg', nrow=8, padding=2)
    else:
        continue"""

count = 0
for data in dataloaders['cat']:
    count += 1
    # get the inputs
    inputs, _ = data

    # wrap them in Variable
    if torch.cuda.is_available():
        inputs = Variable(inputs.cuda())
    else:
        inputs = Variable(inputs)
    
    if count == 1:
        torchvision.utils.save_image(inputs.data, './imgs/POSTER_CAT_IMGS.jpg', nrow=8, padding=2)
    else:
        continue

count = 0
for data in dataloaders['hand']:
    count += 1
    # get the inputs
    inputs, _ = data

    # wrap them in Variable
    if torch.cuda.is_available():
        inputs = Variable(inputs.cuda())
    else:
        inputs = Variable(inputs)
    
    if count == 1:
        torchvision.utils.save_image(inputs.data, './imgs/POSTER_HAND_IMGS.jpg', nrow=8, padding=2)
    else:
        continue

count = 0
for data in dataloaders['flipped']:
    count += 1
    # get the inputs
    inputs, _ = data

    # wrap them in Variable
    if torch.cuda.is_available():
        inputs = Variable(inputs.cuda())
    else:
        inputs = Variable(inputs)
    
    if count == 1:
        torchvision.utils.save_image(inputs.data, './imgs/POSTER_FLIPPED_IMGS.jpg', nrow=8, padding=2)
    else:
        continue
