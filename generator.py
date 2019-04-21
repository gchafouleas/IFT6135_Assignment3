import torch 
import torch.nn as nn
import torch.optim as optim 

import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from torch.autograd import grad
from torch.functional import F

class Generator(nn.Module):
    def __init__(self, width=32, height=32, channels=3, hidden_size=500, 
        latent_size=20, filters=64):

        super(self.__class__, self).__init__()
        #initilization of variables
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.width = width
        self.height = height
        self.channels = channels

        self.preprosse = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, 4*filters, kernel_size=4),
            nn.BatchNorm2d(4*filters),
            nn.ReLU(True),
            nn.ConvTranspose2d(4*filters, 2*filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*filters),
            nn.ReLU(True),
            nn.ConvTranspose2d(2*filters, filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),
            nn.ConvTranspose2d(filters, channels, kernel_size=4, stride=2, padding=1)
        )

        self.sigmoid = nn.Sigmoid()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, z):
        h = self.preprosse(z)
        h = h.view(-1, self.hidden_size, 1, 1)
        x = self.decoder(h)
        x = x.view(-1, self.channels, self.height, self.width)
        x = self.sigmoid(x)
        return x

    def train_model(self,y):
        self.optimizer.zero_grad()
        y = -y.mean()
        y.backward()
        self.optimizer.step()
        return y