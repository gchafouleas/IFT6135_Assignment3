import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.autograd import grad

import numpy as np
from torch.autograd import Variable
from collections import OrderedDict

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
            #nn.Dropout(1- dp_keep_prob)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            #nn.Dropout(1- dp_keep_prob)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            #nn.Dropout(1- dp_keep_prob)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid()
        )

        self.optimizer = optim.SGD(self.parameters(), lr=np.exp(-3))

    def forward(self, inputs):
        x = self.hidden0(inputs)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

    def train(self, x, y, type_loss = "None"):
        self.optimizer.zero_grad()
        x_prediction = self.forward(x)
        y_prediction = self.forward(y)

        loss = 0; 
        if type_loss == "JSD":
            loss = self.loss_JSD(x_prediction, y_prediction)
        elif type_loss == "WD":         
            loss = self.loss_WD(x_prediction, y_prediction, self.Get_z_value(x,y, self.batch_size))
        else : 
            loss = self.loss(x_prediction, y_prediction)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def loss(self, x_pred, y_pred):
        return - (torch.mean(torch.log(x_pred)) + torch.mean(torch.log(y_pred)))

    def loss_JSD(self, x_pred, y_pred):
        return -(torch.log(torch.tensor([[2.]])) + 1/2*torch.mean(torch.log(x_pred) + torch.log((1- y_pred))))

    def loss_WD(self, x_pred, y_pred, norm):
        return -(torch.mean(x_pred) - torch.mean(y_pred) - (self.lamda * torch.mean(((norm -1)**2))))

    def Get_z_value(self, x, y, size):
        a = torch.empty(size,1).uniform_(0,1)
        z_value =  a*x+ (1-a)*y
        z_value.requires_grad = True
        out_interp = self.forward(z_value)
        gradients = grad(outputs=out_interp, inputs=z_value,
                   grad_outputs=torch.ones(out_interp.size()),
                   retain_graph=True, create_graph=True, only_inputs=True)[0]

        # Mean/Expectation of gradients
        gradients = gradients.view(gradients.size(0),  -1)
        gradient_norm = gradients.norm(2, dim=1)
        return gradient_norm