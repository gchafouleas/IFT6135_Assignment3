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
import os

#import data 
directory = "svhn/"
batch_size = 32
torch.manual_seed(1111)
train_loader, valid_loader, test_loader = data.get_data_loader(directory, batch_size)
num_epochs = 1 
discriminator = Discriminator(512)
generator = Generator(512)
model_directory = "gan/"
if torch.cuda.is_available():
    discriminator = discriminator.cuda()
    generator = generator.cuda()
discriminator_updates = 5

def main():
    train_loss_per_epoch = []
    valid_loss_per_epoch = []
    for epoch in range(num_epochs):
        print("epoch "+ str(epoch))
        train_loss = []
        valid_loss = []
        total = 0
        discriminator_correct = 0
        generator_correct = 0
        discriminator.train()
        generator.train()
        for i, data in enumerate(train_loader):
            update = 0
            real_data, targets = data
            N = real_data.size(0)
            #train discriminator
            for i in range(discriminator_updates):
                noise = Variable(torch.randn(N, 100, 1, 1))
                if torch.cuda.is_available():
                    noise = noise.cuda()
                    real_data = real_data.cuda()
                    targets = targets.cuda()
                g_z = generator(noise)
                d_loss, real_prediction, y_prediction = discriminator.train_model(real_data, g_z)
                _, x_predicted = torch.max(real_prediction.data, 1)
                _, y_predicted = torch.max(y_prediction.data, 1)
                discriminator_correct += (x_predicted == targets).sum().item()
                total += targets.size(0)
                generator_correct += (x_predicted == targets).sum().item()
                train_loss.append(d_loss.item())
                update += 1
            #train generator
            if update == discriminator_updates:
                noise = Variable(torch.randn(N, 100, 1, 1))
                if torch.cuda.is_available():
                    noise = noise.cuda()
                g_z = generator(noise)
                output = discriminator(g_z)
                g_loss = generator.train_model(output)
                update = 0
        print("Error dis: ", discriminator_correct/total)
        print("Error gene: ", generator_correct/total)
        train_loss_per_epoch.append(np.mean(train_loss))
        discriminator.eval()
        generator.eval()
        for i, data in enumerate(valid_loader):
            noise = Variable(torch.randn(N, 100, 1, 1))
            if torch.cuda.is_available():
                noise = noise.cuda()
                real_data = real_data.cuda()
                targets = targets.cuda()
            g_z = generator(noise)
            fake = discriminator(g_z)
            real = discriminator(real_data)
            d_loss= discriminator.loss(real, fake)
            valid_loss.append(d_loss.item())
            _, x_predicted = torch.max(real_prediction.data, 1)
            _, y_predicted = torch.max(y_prediction.data, 1)
            discriminator_correct += (x_predicted == targets).sum().item()
            total += targets.size(0)
            generator_correct += (x_predicted == targets).sum().item()
        valid_loss_per_epoch.append(np.mean(valid_loss))
        #saving model for each epoch
        torch.save(discriminator.state_dict(), os.path.join(model_directory + "models/", str(epoch)+'_model.pt'))

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
