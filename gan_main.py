import torch 
import torch.nn as nn
import torch.optim as optim 

import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from Gan_model import Discriminator
from Gan_model import Generator
import classify_svhn as data
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch model')
parser.add_argument('--eval_mode', type=str, default='Train',
                    help='eval mode to use: Train or Test')

args = parser.parse_args()
#import data 
directory = "svhn/"
batch_size = 32
torch.manual_seed(1111)
train_loader, valid_loader, test_loader = data.get_data_loader(directory, batch_size)
num_epochs = 100 
discriminator = Discriminator(32, 50, batch_size, 0.3)
generator = Generator(32)
model_directory = "gan/"
if torch.cuda.is_available():
    discriminator = discriminator.cuda()
    generator = generator.cuda()
discriminator_updates = 6

def main():
    if args.eval_mode == "Train":
        train_loss_per_epoch = []
        valid_loss_per_epoch = []
        for epoch in range(num_epochs):
            print("epoch "+ str(epoch))
            train_loss = []
            valid_loss = []
            discriminator.train()
            generator.train()
            discriminator.zero_grad()
            generator.zero_grad()
            update = 0
            for i, data in enumerate(train_loader):
                real_data, targets = data
                N = real_data.size(0)
                #train discriminator
                noise = Variable(torch.randn(N, 100,1,1))
                if torch.cuda.is_available():
                    noise = noise.cuda()
                    real_data = real_data.cuda()
                    targets = targets.cuda()
                g_z = generator(noise)
                d_loss, real_prediction, y_prediction = discriminator.train_model(real_data, g_z)
                train_loss.append(d_loss.item())
                #train generator
                if update == discriminator_updates:
                    discriminator.zero_grad()
                    generator.zero_grad()
                    noise = Variable(torch.randn(N, 100,1,1))
                    if torch.cuda.is_available():
                        noise = noise.cuda()
                    g_z = generator(noise)
                    output = discriminator(g_z)
                    g_loss = generator.train_model(output)
                    update = 0
                update += 1
            train_loss_per_epoch.append(np.mean(train_loss))
            print("train loss: ", np.mean(train_loss))
            discriminator.eval()
            generator.eval()
            for i, data in enumerate(valid_loader):
                real_data, targets = data
                N = real_data.size(0)
                noise = Variable(torch.randn(N, 100,1,1))
                if torch.cuda.is_available():
                    noise = noise.cuda()
                    real_data = real_data.cuda()
                    targets = targets.cuda()
                g_z = generator(noise)
                fake = discriminator(g_z)
                real = discriminator(real_data)
                d_loss= torch.mean(real) - torch.mean(fake)
                valid_loss.append(d_loss.item())
            valid_loss_per_epoch.append(np.mean(valid_loss))
            print("valid loss: ", np.mean(valid_loss))
            #saving model for each epoch
            torch.save(generator.state_dict(), os.path.join(model_directory + "models/", str(epoch)+'_generator.pt'))

        plt.plot(train_loss_per_epoch, 'o-')
        plt.plot(valid_loss_per_epoch, 'o-')
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("loss vs epoch")
        plt.legend(labels = ["train", "valid"])
        plt.savefig(model_directory + 'loss_epoch.png', bbox_inches='tight')
        plt.clf()

    if args.eval_mode == "Test":
        to_img = ToPILImage()
        for i, data in enumerate(test_loader):
            real_data, targets = data
            to_img(real_data.normal_())

if __name__=='__main__':
    main()         
