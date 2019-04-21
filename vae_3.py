import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

import numpy as np
import math

class VAE(nn.Module):
    def __init__(self, width=32, height=32, channels=3, 
        latent_size=100):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.width = width
        self.height = height
        self.channels = channels

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 2*64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*64),
            nn.ReLU(True),
            nn.Conv2d(2*64, 4*64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4*64),
            nn.ReLU(True),
            nn.Conv2d(4*64, 500, kernel_size=4),
            nn.BatchNorm2d(500),
            nn.ReLU(True)
        )

        self.mu = nn.Linear(500, latent_size)
        self.log_sigma = nn.Linear(500, latent_size)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def loss_function(self, decode_x, x, mu, logvar):
        decode_x = decode_x.view(decode_x.size(0), -1)
        x = x.view(x.size(0), -1)
        BCE = -torch.sum(x*torch.log(torch.clamp(decode_x, min=1e-10))+
            (1-x)*torch.log(torch.clamp(1-decode_x, min=1e-10)), 1)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        # Normalise by same number of elements as in reconstruction
        loss = torch.mean(BCE + KLD)

        return loss

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, 500)
        mu = self.mu(h)
        logvar = self.log_sigma(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar