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
import math
from PIL import Image
import warnings
from scipy.stats import multivariate_normal
from vae import VAE, ShallowVAE, BasicBlock


parser = argparse.ArgumentParser(description='PyTorch distribution params')
parser.add_argument('--path', type=str,
                    help='path to .pth file in models/')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
args = parser.parse_args()

BATCH_SIZE = args.batch_size

path_in = '../InDistribution/'
path_out = '../OutDistribution/'
path_cat = '../PetImages/train/'
path_hand = '../Bone/valid/'
kwargs = {'num_workers': 3, 'pin_memory': True}

"""simple_transform = transforms.Compose([transforms.Resize((224,224))
                                       ,transforms.ToTensor(), transforms.Normalize([0.48829153, 0.45526633, 0.41688013],[0.25974154, 0.25308523, 0.25552085])])"""
simple_transform = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.Grayscale(num_output_channels=3),  #HACK
                                       transforms.ToTensor()])
cat_transform = transforms.Compose([transforms.Resize((224,224)),
                                       #transforms.Grayscale(num_output_channels=3),  #HACK
                                       transforms.ToTensor()])
in_distr = ImageFolder(path_in+'train/',simple_transform)
in_distr_data_gen = torch.utils.data.DataLoader(in_distr,shuffle=True,batch_size=BATCH_SIZE,num_workers=kwargs['num_workers'])

in_distr_val = ImageFolder(path_in+'valid/',simple_transform)
in_distr_val_data_gen = torch.utils.data.DataLoader(in_distr_val,shuffle=True,batch_size=BATCH_SIZE,num_workers=kwargs['num_workers'])

out_distr = ImageFolder(path_out,simple_transform)
out_distr_data_gen = torch.utils.data.DataLoader(out_distr,shuffle=True,batch_size=BATCH_SIZE,num_workers=kwargs['num_workers'])

cat_distr = ImageFolder(path_cat,cat_transform)
cat_distr_data_gen = torch.utils.data.DataLoader(cat_distr,shuffle=True,batch_size=BATCH_SIZE,num_workers=kwargs['num_workers'])

hand_distr = ImageFolder(path_hand,simple_transform)
hand_distr_data_gen = torch.utils.data.DataLoader(hand_distr,shuffle=True,batch_size=BATCH_SIZE,num_workers=kwargs['num_workers'])

dataset_sizes = {'in_distr':len(in_distr_data_gen.dataset),
    'in_distr_val':len(in_distr_val_data_gen.dataset), 
    'out_distr':len(out_distr_data_gen.dataset),
    'cat_distr':len(cat_distr_data_gen.dataset),
    'hand_distr':len(hand_distr_data_gen.dataset)  }
dataloaders = {'in_distr':in_distr_data_gen,
    'in_distr_val':in_distr_val_data_gen, 
    'out_distr':out_distr_data_gen,
    'cat_distr':cat_distr_data_gen,
    'hand_distr':hand_distr_data_gen}


model = VAE(BasicBlock, [2, 2, 2, 2], latent_variable_size=500, nc=3, ngf=224, ndf=224, is_cuda=True)
model.load_state_dict(torch.load("models/"+args.path))
model.cuda()
model.eval()

#Epoch_157_Train_loss_14.0268_Test_loss_17.2915.pth grayscale
#Epoch_169_Train_loss_13.8677_Test_loss_17.0471.pth no grayscale
#Epoch_126_Train_loss_6.9641_Test_loss_9.2266.pth no gs, 0.1 KL

def in_distribution_params():
    means = None
    std_devs = None
    for data in dataloaders['in_distr']:
        # get the inputs
        inputs, _ = data

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        _, mu, logvar = model(inputs)
        std = logvar.mul(0.5).exp_()
        mu = mu.detach().cpu().numpy()
        std = std.detach().cpu().numpy()
        if means is None:
            means = mu
            std_devs = std
            continue
        means = np.concatenate((means, mu))
        std_devs = np.concatenate((std_devs, std))
    avg_mu = np.average(means, axis=0) #now a (500,) vector
    avg_std = np.average(std_devs, axis=0) #now a (500,) vector
    avg_var_mu = np.var(means, axis=0)  #(500,)
    avg_var = np.square(avg_std)
    return (avg_mu, avg_var, avg_var_mu)

def out_distribution_params():
    means = None
    std_devs = None
    for data in dataloaders['out_distr']:
        # get the inputs
        inputs, _ = data

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        _, mu, logvar = model(inputs)
        std = logvar.mul(0.5).exp_()
        mu = mu.detach().cpu().numpy()
        std = std.detach().cpu().numpy()
        if means is None:
            means = mu
            std_devs = std
            continue
        means = np.concatenate((means, mu))
        std_devs = np.concatenate((std_devs, std))
    avg_mu = np.average(means, axis=0) #now a (500,) vector
    avg_std = np.average(std_devs, axis=0) #now a (500,) vector
    avg_var_mu = np.var(means, axis=0)  #(500,)
    avg_var = np.square(avg_std)
    return (avg_mu, avg_var, avg_var_mu)

def cat_distribution_params(): #cats training set, not val set
    means = None
    std_devs = None
    for data in dataloaders['cat_distr']:
        # get the inputs
        inputs, _ = data

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        _, mu, logvar = model(inputs)
        std = logvar.mul(0.5).exp_()
        mu = mu.detach().cpu().numpy()
        std = std.detach().cpu().numpy()
        if means is None:
            means = mu
            std_devs = std
            continue
        means = np.concatenate((means, mu))
        std_devs = np.concatenate((std_devs, std))
    avg_mu = np.average(means, axis=0) #now a (500,) vector
    avg_std = np.average(std_devs, axis=0) #now a (500,) vector
    avg_var_mu = np.var(means, axis=0)  #(500,)
    avg_var = np.square(avg_std)
    return (avg_mu, avg_var, avg_var_mu)

def hand_distribution_params(): #hand test set
    means = None
    std_devs = None
    for data in dataloaders['hand_distr']:
        # get the inputs
        inputs, _ = data

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        _, mu, logvar = model(inputs)
        std = logvar.mul(0.5).exp_()
        mu = mu.detach().cpu().numpy()
        std = std.detach().cpu().numpy()
        if means is None:
            means = mu
            std_devs = std
            continue
        means = np.concatenate((means, mu))
        std_devs = np.concatenate((std_devs, std))
    avg_mu = np.average(means, axis=0) #now a (500,) vector
    avg_std = np.average(std_devs, axis=0) #now a (500,) vector
    avg_var_mu = np.var(means, axis=0)  #(500,)
    avg_var = np.square(avg_std)
    return (avg_mu, avg_var, avg_var_mu)

"""def in_distribution_val_params():
    means = None
    std_devs = None
    for data in dataloaders['in_distr_val']:
        # get the inputs
        inputs, _ = data

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        _, mu, logvar = model(inputs)
        std = logvar.mul(0.5).exp_()
        mu = mu.detach().cpu().numpy()
        std = std.detach().cpu().numpy()
        if means is None:
            means = mu
            std_devs = std
            continue
        means = np.concatenate((means, mu))
        std_devs = np.concatenate((std_devs, std))
    return (means, std_devs)"""

"""def out_distribution_params():
    means = None
    std_devs = None
    for data in dataloaders['out_distr']:
        # get the inputs
        inputs, _ = data

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        _, mu, logvar = model(inputs)
        std = logvar.mul(0.5).exp_()
        mu = mu.detach().cpu().numpy()
        std = std.detach().cpu().numpy()
        if means is None:
            means = mu
            std_devs = std
            continue
        means = np.concatenate((means, mu))
        std_devs = np.concatenate((std_devs, std))
    return (means, std_devs)"""

in_avg_mu, in_avg_var, in_avg_var_mu = in_distribution_params()
out_avg_mu, out_avg_var, out_avg_var_mu = out_distribution_params()
cat_avg_mu, cat_avg_var, cat_avg_var_mu = cat_distribution_params()
hand_avg_mu, hand_avg_var, hand_avg_var_mu = hand_distribution_params()
import pdb
pdb.set_trace()
#val_means, val_std_devs = in_distribution_val_params()
#out_means, out_std_devs = out_distribution_params()
#import pdb
#pdb.set_trace()

