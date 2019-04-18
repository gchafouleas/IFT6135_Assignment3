import torch 
import torch.nn as nn
import torch.optim as optim 

import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from torch.autograd import grad
from torch.functional import F

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

        self.conv7 = nn.Conv2d(16, 3, 3, padding=4)

        self.optimizer = optim.Adam(self.parameters())

    def forward(self, inputs):
        h1 = self.elu4(self.fc2(inputs))
        h1 = h1[:,:,None,None]
        h2 = self.sample1(self.elu5(self.conv4(h1)))
        h3 = self.sample2(self.elu6(self.conv5(h2)))
        h4 = self.elu7(self.conv6(h3))
        h5 = self.conv7(h4)
        Sigmoid = nn.Sigmoid()
        return Sigmoid(h5)

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