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
kwargs = {'num_workers': 3, 'pin_memory': True}

simple_transform = transforms.Compose([transforms.Resize((224,224))
                                       ,transforms.ToTensor(), transforms.Normalize([0.48829153, 0.45526633, 0.41688013],[0.25974154, 0.25308523, 0.25552085])])
in_distr = ImageFolder(path_in+'train/',simple_transform)
in_distr_data_gen = torch.utils.data.DataLoader(in_distr,shuffle=True,batch_size=BATCH_SIZE,num_workers=kwargs['num_workers'])

in_distr_val = ImageFolder(path_in+'valid/',simple_transform)
in_distr_val_data_gen = torch.utils.data.DataLoader(in_distr_val,shuffle=True,batch_size=BATCH_SIZE,num_workers=kwargs['num_workers'])

out_distr = ImageFolder(path_out,simple_transform)
out_distr_data_gen = torch.utils.data.DataLoader(out_distr,shuffle=True,batch_size=BATCH_SIZE,num_workers=kwargs['num_workers'])

dataset_sizes = {'in_distr':len(in_distr_data_gen.dataset),'in_distr_val':len(in_distr_val_data_gen.dataset), 'out_distr':len(out_distr_data_gen.dataset)}
dataloaders = {'in_distr':in_distr_data_gen,'in_distr_val':in_distr_val_data_gen, 'out_distr':out_distr_data_gen}


model = VAE(BasicBlock, [2, 2, 2, 2], latent_variable_size=500, nc=3, ngf=224, ndf=224, is_cuda=True)
model.load_state_dict(torch.load("models/"+args.path))
model.cuda()
model.eval()


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
        #import pdb
        #pdb.set_trace()
        mu = mu.detach().cpu().numpy()
        std = std.detach().cpu().numpy()
        if means is None:
            means = mu
            std_devs = std
            continue
        means = np.concatenate((means, mu))
        std_devs = np.concatenate((std_devs, std))
    avg_mu = np.average(means, axis=0)
    avg_std = np.average(std_devs, axis=0)
    return (avg_mu, avg_std)

def in_distribution_val_params():
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
    return (means, std_devs)

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
    return (means, std_devs)

def compute_likelihood(x, mu, std_dev):
    return multivariate_normal.pdf(x, mu, std_dev)

avg_mu, avg_std = in_distribution_params()
val_means, val_std_devs = in_distribution_val_params()
out_means, out_std_devs = out_distribution_params()
import pdb
pdb.set_trace()


