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


if not os.path.exists(model_directory):
    print("creating VAE directory")
    os.mkdir(model_directory)
    os.mkdir(model_directory + '/imgs')
    os.mkdir(model_directory + '/models')

MSE = nn.MSELoss(reduction='sum').cuda()

generator = Generator(latent_size=100)

model = VAE(latent_size=100)
if torch.cuda.is_available():
    model.cuda()
    generator.cuda()

optimizer_encoder = optim.Adam(model.parameters(), lr=1e-4)
optimizer_generator = optim.Adam(generator.parameters(), lr=1e-4)

def vae_loss(decode_x, x, mu, logvar):
    loss = MSE(decode_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
    loss = torch.mean(loss + KLD)

    return loss

def train_model():
    model.train()
    generator.train()
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
    generator.eval()
    valid_loss = []
    for i, data in enumerate(valid_loader):
        real_data, targets = data
        real_data = Variable(real_data)
        if torch.cuda.is_available():
            real_data = real_data.cuda()
        z, mu, logvar = model(real_data)
        decode_x = generator(z)
        loss = vae_loss(decode_x, real_data, mu, logvar)
        torchvision.utils.save_image(real_data.data, 'vae/imgs/'+ str(epoch) + '_real_image.png', nrow=8, padding=2)
        torchvision.utils.save_image(decode_x, 'vae/imgs/'+ str(epoch) + '_decoded_image.png', nrow=8, padding=2)
        valid_loss.append(loss.item())

    return np.mean(valid_loss)

def main():
    for epoch in range(num_epochs):
        print("epoch : ", epoch)
        train_loss_per_epoch = []
        valid_loss_per_epoch = []
        train_epoch_loss, mu, logvar = train_model()
        train_loss_per_epoch.append(train_epoch_loss)
        valid_epoch_loss = valid_model(epoch)
        valid_loss_per_epoch.append(valid_epoch_loss)
        print("train loss: ", train_epoch_loss)
        print("valid loss: ", valid_epoch_loss)
        noise = Variable(torch.randn(32, 100)).cuda()
        image = generator(noise)
        torchvision.utils.save_image(image, 'vae/'+ str(epoch) + 'image.png', nrow=8, padding=2)
        torch.save(generator.state_dict(), os.path.join(model_directory + "models/", str(epoch)+'_decoder.pt'))

    plt.plot(train_loss_per_epoch, 'o-')
    plt.plot(valid_loss_per_epoch, 'o-')
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("loss vs epoch")
    plt.legend(labels = ["train", "valid"])
    plt.savefig(model_directory + 'loss_epoch.png', bbox_inches='tight')
    plt.clf()

if __name__=='__main__':
    main() 