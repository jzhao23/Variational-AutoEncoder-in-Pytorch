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

def load_npy(filename):
    data = np.load("../ood_data/" + filename)
    print(filename + ": ", data)
    print("data mean: ", np.average(data))
    print("data median: ", np.median(data))
    print("data min: ", np.min(data))
    print("data max: ", np.max(data))

load_npy("in.npy")
load_npy("out.npy")
load_npy("flipped.npy")
load_npy("cat.npy")
load_npy("hand.npy")
