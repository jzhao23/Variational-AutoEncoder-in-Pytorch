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
    print(filename)
    print("data mean: ", np.average(data))
    print("data median: ", np.median(data))
    print("data min: ", np.min(data))
    print("data max: ", np.max(data))

def threshold_test(filename, threshold):
    print(threshold)
    print(filename)
    data = np.load("../ood_data/" + filename)
    total = len(data)
    in_d = 0
    out_d = 0
    for x in data:
        if x < threshold:
            out_d += 1
        else:
            in_d += 1
    print("percent in  distr: ", in_d/(total * 1.0))
    print("percent out  distr: ", out_d/(total * 1.0))

    

load_npy("in.npy")
load_npy("out.npy")
load_npy("flipped.npy")
load_npy("cat.npy")
load_npy("hand.npy")


threshold_test("in.npy", 0.001)
threshold_test("out.npy", 0.001)
threshold_test("flipped.npy", 0.001)
threshold_test("cat.npy", 0.001)
threshold_test("hand.npy", 0.001)
print("="*80)

threshold_test("in.npy", 0.0015)
threshold_test("out.npy", 0.0015)
threshold_test("flipped.npy", 0.0015)
threshold_test("cat.npy", 0.0015)
threshold_test("hand.npy", 0.0015)
print("="*80)

threshold_test("in.npy", 0.002)
threshold_test("out.npy", 0.002)
threshold_test("flipped.npy", 0.002)
threshold_test("cat.npy", 0.002)
threshold_test("hand.npy", 0.002)
print("="*80)

threshold_test("in.npy", 0.0025)
threshold_test("out.npy", 0.0025)
threshold_test("flipped.npy", 0.0025)
threshold_test("cat.npy", 0.0025)
threshold_test("hand.npy", 0.0025)
print("="*80)
threshold_test("in.npy", 0.003)
threshold_test("out.npy", 0.003)
threshold_test("flipped.npy", 0.003)
threshold_test("cat.npy", 0.003)
threshold_test("hand.npy", 0.003)
print("="*80)

threshold_test("in.npy", 0.004)
threshold_test("out.npy", 0.004)
threshold_test("flipped.npy", 0.004)
threshold_test("cat.npy", 0.004)
threshold_test("hand.npy", 0.004)
print("="*80)

threshold_test("in.npy", 0.005)
threshold_test("out.npy", 0.005)
threshold_test("flipped.npy", 0.005)
threshold_test("cat.npy", 0.005)
threshold_test("hand.npy", 0.005)
print("="*80)

threshold_test("in.npy", 0.006)
threshold_test("out.npy", 0.006)
threshold_test("flipped.npy", 0.006)
threshold_test("cat.npy", 0.006)
threshold_test("hand.npy", 0.006)
print("="*80)