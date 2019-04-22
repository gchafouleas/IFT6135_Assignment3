"""Variational autoencoder model implemented for Q2 (MNIST Binary Dataset."""

from torchvision.datasets import utils
import torch.utils.data as data_utils
import torch
import os
import numpy as np
from torch import nn
from torch.nn.modules import upsampling
from torch.functional import F
from torch.optim import Adam
from torch.autograd import Variable
import matplotlib.pyplot as plt


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # encoder
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 256, 5)
        self.elu3 = nn.ELU()
        self.fc1 = nn.Linear(256, 100)
        self.fc12 = nn.Linear(256, 100)

        # decoder
        self.fc2 = nn.Linear(100, 256)
        self.elu4 = nn.ELU()

        self.conv4 = nn.Conv2d(256, 64, 5, padding=4)
        self.elu5 = nn.ELU()
        self.sample1 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv5 = nn.Conv2d(64, 32, 3, padding=2)
        self.elu6 = nn.ELU()
        self.sample2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv6 = nn.Conv2d(32, 16, 3, padding=2)
        self.elu7 = nn.ELU()

        self.conv7 = nn.Conv2d(16, 1, 3, padding=2)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):

        h1 = self.pool1(self.elu1(self.conv1(x)))
        h2 = self.pool2(self.elu2(self.conv2(h1)))
        h3 = self.elu3(self.conv3(h2))
        mu = self.fc1(h3.squeeze())
        sigma = self.fc12(h3.squeeze())

        return mu, sigma

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar) 
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        z = torch.mul(eps, std) + mu
        return z

    def decode(self, z):
        h1 = self.elu4(self.fc2(z))
        h1 = h1[:,:,None,None]
        h2 = self.sample1(self.elu5(self.conv4(h1)))
        h3 = self.sample2(self.elu6(self.conv5(h2)))
        h4 = self.elu7(self.conv6(h3))
        h5 = self.conv7(h4)
        out = self.sigmoid(h5)
        return out

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparam(mu, sigma)
        x_recon = self.decode(z)
        return x_recon, mu, sigma
