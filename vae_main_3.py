import torch 
import torch.nn as nn
import torch.optim as optim 

import numpy as np
from torch.autograd import Variable
from torch.autograd import grad
from vae_3 import VAE
import classify_svhn as data
import argparse
import torchvision
import os

parser = argparse.ArgumentParser(description='PyTorch model')
parser.add_argument('--eval_mode', type=str, default='Train',
                    help='eval mode to use: Train or Test')

args = parser.parse_args()
#import data 
directory = "svhn/"
model_directory = "vae/"
batch_size = 32
torch.manual_seed(1111)
train_loader, valid_loader, test_loader = data.get_data_loader(directory, batch_size)
num_epochs = 100

binary_loss = nn.BCELoss(reduction = 'sum')
model = VAE(channels=3, width=32, height=32, latent_variable_size=100)
if torch.cuda.is_available():
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def vae_loss(decode_x, x, mu, logvar):
    loss = binary_loss(decode_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return loss + KLD

def train_model():
    model.train()
    train_loss = []
    for i, data in enumerate(train_loader):
        real_data, targets = data
        real_data = Variable(real_data)
        if torch.cuda.is_available():
            real_data = real_data.cuda()
        optimizer.zero_grad()
        decode_x, mu, logvar = model(real_data)
        loss = vae_loss(decode_x, real_data, mu, logvar)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
    return np.mean(train_loss)

def valid_model(epoch):
    model.eval()
    valid_loss = []
    for i, data in enumerate(valid_loader):
        real_data, targets = data
        real_data = Variable(real_data)
        if torch.cuda.is_available():
            real_data = real_data.cuda()
        decode_x, mu, logvar = model(real_data)
        loss = vae_loss(decode_x, real_data, mu, logvar)
        valid_loss.append(loss.item())
        torchvision.utils.save_image(real_data.data, 'vae/imgs/Epoch_{}_data.jpg'.format(epoch), nrow=8, padding=2)
        torchvision.utils.save_image(decode_x.data, 'vae/imgs/Epoch_{}_recon.jpg'.format(epoch), nrow=8, padding=2)

def main():
    for epoch in range(num_epochs):
        print("epoch : ", epoch)
        train_epoch_loss = train_model()
        valid_epoch_loss = valid_model(epoch)
        print("train ",train_epoch_loss)
        torch.save(model.state_dict(), os.path.join(model_directory + "models/", str(epoch)+'_decoder.pt'))


if __name__=='__main__':
    main() 