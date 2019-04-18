import torch 
import torch.nn as nn
import torch.optim as optim 

import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from torch.autograd import grad

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, dp_keep_prob):

        super(Discriminator, self).__init__()
        #initilization of variables
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dp_keep_prob = dp_keep_prob
        self.lamda = 10
        #Initialize hidden layers
        self.hidden0 = nn.Sequential( 
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(1- dp_keep_prob)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(1- dp_keep_prob)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(1- dp_keep_prob)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters())

    def forward(self, inputs):
        x = self.hidden0(inputs)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

    def train_model(self, x, y):

        self.optimizer.zero_grad()
        x_prediction = self.forward(x)
        y_prediction = self.forward(y)

        one = torch.FloatTensor([1])
        mone = one * -1
        if torch.cuda.is_available():
            one = one.cuda()
            mone = mone.cuda()

        gradient_penalty = self.Get_z_value(x,y)
        gradient_penalty.backward()
        x_prediction = x_prediction.mean()
        y_prediction = y_prediction.mean()
        y_prediction.backward(one)
        x_prediction.backward(mone)
        D_cost = y_prediction - x_prediction + gradient_penalty
        Wasserstein_D = x_prediction - y_prediction
        self.optimizer.step()
        return Wasserstein_D, x_prediction, y_prediction

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

        return ((gradients.norm(2, dim=1) - 1)**2).mean() * self.lamda

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

        self.optimizer = optim.Adam(self.parameters())

    def forward(self, inputs):
        return self.main(inputs)

    def train_model(self,y):

        self.optimizer.zero_grad()
        one = torch.FloatTensor([1])
        mone = one * -1
        if torch.cuda.is_available():
            mone = mone.cuda()
        y = y.mean()
        y.backward(mone)
        cost = -y
        self.optimizer.step()
        return cost

    def safe_mean(self, input):
        input = input.mean(dim = 0)
        input = input.mean(dim=0)
        return input.mean()