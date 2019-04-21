import torch 
import torch.nn as nn
import torch.optim as optim 

import numpy as np
from torch.autograd import Variable
from torch.autograd import grad
from vae_3 import VAE
from generator import Generator 
import classify_svhn as data
import matplotlib.pyplot as plt
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

binary_loss = nn.BCELoss(reduction='sum').cuda()
generator = Generator(width=32, height=32, channels=3, hidden_size=500, latent_size=100,
        filters=64)
model = VAE(width=32, height=32, channels=3, latent_size=100)
if torch.cuda.is_available():
    model.cuda()
    generator.cuda()
optimizer_encoder = optim.Adam(model.parameters(), lr=1e-4)
optimizer_generator = optim.Adam(generator.parameters(), lr=1e-4)

def vae_loss(decode_x, x, mu, logvar):
    decode_x = decode_x.view(decode_x.size(0), -1)
    x = x.view(x.size(0), -1)
    BCE = -torch.sum(x*torch.log(torch.clamp(decode_x, min=1e-10))+
        (1-x)*torch.log(torch.clamp(1-decode_x, min=1e-10)), 1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        # Normalise by same number of elements as in reconstruction
    loss = torch.mean(BCE + KLD)

    return loss

def train_model():
    model.train()
    train_loss = []
    for i, data in enumerate(train_loader):
        real_data, targets = data
        real_data = Variable(real_data)
        if torch.cuda.is_available():
            real_data = real_data.cuda()
        optimizer_generator.zero_grad()
        optimizer_encoder.zero_grad()
        z, mu, logvar = model(real_data)
        decode_x = generator(z)
        loss = vae_loss(decode_x, real_data, mu, logvar)
        loss.backward()
        train_loss.append(loss.item())
        optimizer_encoder.step()
        optimizer_generator.step()
    return np.mean(train_loss), mu, logvar

def valid_model(epoch):
    model.eval()
    valid_loss = []
    for i, data in enumerate(valid_loader):
        real_data, targets = data
        real_data = Variable(real_data)
        if torch.cuda.is_available():
            real_data = real_data.cuda()
        z, mu, logvar = model(real_data)
        decode_x = generator(z)
        loss = vae_loss(decode_x, real_data, mu, logvar)
        valid_loss.append(loss.item())

def main():
    for epoch in range(num_epochs):
        print("epoch : ", epoch)
        train_epoch_loss, mu, logvar = train_model()

        noise = Variable(torch.randn(32, 100)).cuda()
        image = generator(noise)
        torchvision.utils.save_image(image, 'vae/'+ str(epoch) + 'image.png', nrow=8, padding=2)

        valid_epoch_loss = valid_model(epoch)
        print("train ",train_epoch_loss)
        torch.save(model.state_dict(), os.path.join(model_directory + "models/", str(epoch)+'_decoder.pt'))


if __name__=='__main__':
    main() 