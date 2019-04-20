import torch 
import torch.nn as nn
import torch.optim as optim 

import numpy as np
from torch.autograd import Variable
from torch.autograd import grad

class VAE(nn.Module):
    def __init__(self, channels, width, height, latent_variable_size):
        super(VAE, self).__init__()

        self.channels = channels
        self.width = width
        self.height = height
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(3,height ,4, 2, 1)
        self.bn1 = nn.BatchNorm2d(height)

        self.e2 = nn.Conv2d(height, height*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(height*2)

        self.e3 = nn.Conv2d(height*2, height*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(height*4)

        self.e4 = nn.Conv2d(height*4, height*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(height*8)

        self.e5 = nn.Conv2d(height*8, height*8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(height*8)

        self.fc1 = nn.Linear(height*8, latent_variable_size)
        self.fc2 = nn.Linear(height*8, latent_variable_size)

        self.d1 = nn.Linear(latent_variable_size, width*8*2*4*4)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.d2 = nn.Linear(latent_variable_size, latent_variable_size)

        # decoder the same as the generator from the gan
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(width*8*2, width*8, 3, 1)
        self.bn6 = nn.BatchNorm2d(width*8, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(width*8, width*4, 3, 1)
        self.bn7 = nn.BatchNorm2d(width*4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(width*4, width, 3, 1)
        self.bn8 = nn.BatchNorm2d(width, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=1)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(width, channels, 3, 1)


    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(h5.size(0), self.width*8)
        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        #z = self.d2(z)
        #z = z.view(z.size(0), self.latent_variable_size, 1, 1)
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.width*8*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h5 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5))))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.channels, self.width, self.height))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.channels, self.height, self.height))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar