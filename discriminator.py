import torch 
import torch.nn as nn
import torch.optim as optim 

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

    def train(self, x, y, type_loss = "JSD"):

        self.optimizer.zero_grad()
        x_prediction = self.forward(x)
        y_prediction = self.forward(y)

        loss = 0; 
        if type_loss == "JSD":
            loss = self.loss_JSD(x_prediction, y_prediction)
        #else:
        #    a = np.random.uniform(low=0.0, high=1.0,)
        #    z_value =  a*real_data + (1-a)*fake_date
        #    out = self.forward(z_value)
        #    out.backward()
        #    loss = self.loss_WD(real_data_prediction, fake_data_prediction, z.z_value.grad.data.norm(2))
        loss.backward()
        self.optimizer.step()

    def loss_JSD(self, x_pred, y_pred):
        return -(torch.log(torch.tensor([[2.]])) + 1/2*torch.mean(torch.log(x_pred)) + 1/2*torch.mean(torch.log((1- y_pred))))

    def loss_WD(self, real, fake, norm):
        return np.mean(real) - np.mean(fake) + self.lamda * (norm -1)**2
