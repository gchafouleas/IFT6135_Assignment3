import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable

import numpy as np
import math

class VAE(nn.Module):
    def __init__(self, 
        latent_size=100):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 500, kernel_size=4),
            nn.BatchNorm2d(500),
            nn.ReLU(True)
        )

        self.mu = nn.Linear(500, latent_size)
        self.log_sigma = nn.Linear(500, latent_size)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, 500)
        mu = self.mu(h)
        logvar = self.log_sigma(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar