import torch 
import torch.nn as nn
import torch.optim as optim 

import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from torch.autograd import grad

class Discriminator(nn.Module):
    def __init__(self, batch_size):

        super(Discriminator, self).__init__()
        #initilization of variables
        self.batch_size = batch_size
        self.lamda = 10
        #Initialize hidden layers
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 512, 2),
        )

        self.mlp = nn.Sequential(
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-2)

    def forward(self, inputs):
        return self.mlp(self.extract_features(inputs))

    def extract_features(self, x):
        return self.conv_stack(x)[:, :, 0, 0]

    def train_model(self, x, y):

        self.optimizer.zero_grad()
        x_prediction = self.forward(x)
        y_prediction = self.forward(y)
        loss = 0
        loss = self.loss(x_prediction, y_prediction)
        loss.backward()
        self.optimizer.step()
        return loss, x_prediction, y_prediction

    def loss(self, x_pred, y_pred):
        return -(torch.mean(x_pred) - torch.mean(y_pred))

    def safe_mean(self, input):
        input = input.mean(dim = 0)
        input = input.mean(dim=0)
        return input.mean()

    def Get_z_value(self, x, y):
        a = torch.empty(x.shape).uniform_(0,1)
        if torch.cuda.is_available():
            a = a.cuda()
        z =  a*x+ (1-a)*y
        z_value = Variable(z, requires_grad=True)
        z_value.cuda()
        out_interp = self.forward(z_value)
        gradients = grad(outputs=out_interp, inputs=z_value,
                   grad_outputs=torch.ones(out_interp.size()).cuda(),
                   retain_graph=True, create_graph=True, only_inputs=True)[0]

        return gradients.view(gradients.size(0),  -1).norm(2, dim=1)

class Generator(nn.Module):
    def __init__(self, batch_size):

        super(Generator, self).__init__()
        #initilization of variables
        self.batch_size = batch_size
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( 64 * 2, 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-2)

    def forward(self, inputs):
        return self.main(inputs)

    def train_model(self,y):

        self.optimizer.zero_grad()
        loss = self.loss(y)
        loss.backward()
        self.optimizer.step()
        return loss

    def safe_mean(self, input):
        input = input.mean(dim = 0)
        input = input.mean(dim=0)
        return input.mean()

    def loss(self, y_pred):
        return -self.safe_mean(torch.log(y_pred))