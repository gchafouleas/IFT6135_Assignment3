import torch 
import torch.nn as nn

import numpy as np
from torch.autograd import Variable
from collections import OrderedDict

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, batch_size, num_layers, dp_keep_prob):

        super(Discriminator, self).__init__()
        #initilization of variables
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob

        #Initialize hidden layers
        self.hidden0 = nn.Sequential( 
            nn.Linear(input_size, hidden_size),
            nn.ReLu()
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLu()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLu()
        )
        self.out = nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, inputs, hidden):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x