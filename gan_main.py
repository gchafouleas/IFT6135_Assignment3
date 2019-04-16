import torch 
import torch.nn as nn
import torch.optim as optim 

import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from Gan_model import Discriminator
from Gan_model import Generator
import classify_svhn as data

#import data 
directory = "svhn/"
batch_size = 512
train_loader, valid_loader, test_loader = data.get_data_loader(directory, batch_size)
print("done downloader images")
num_epochs = 50 
discriminator = Discriminator(512)
generator = Generator(512)
discriminator_updates = 5

def main():
    loss_per_epoch = []
    for epoch in range(num_epochs):
        print("epoch "+ str(epoch))
        loss = []
        for i, data in enumerate(train_loader):
            update = 0
            real_data, targets = data
            N = real_data.size(0)
            #train discriminator
            for i in range(discriminator_updates):
                noise = Variable(torch.randn(N, 100, 1, 1))
                g_z = generator(noise)
                d_loss = discriminator.train(real_data, g_z)
                print(d_loss)
                loss.append(d_loss)
                update += 1

            #train generator
            if update == discriminator_updates:
                noise = Variable(torch.randn(N, 100, 1, 1))
                g_z = generator(noise)
                output = discriminator(g_z)
                g_loss = generator.train(output)
                update = 0
        print("loss per epoch: "+ str(np.mean(loss)))
        loss_per_epoch.append(np.mean(loss))

    plt.plot(loss_per_epoch, 'o-')
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("loss vs epoch")
    #plt.savefig(directory + '_JSD_phi.png', bbox_inches='tight')
    plt.show()
    plt.clf()        
if __name__=='__main__':
    main()         
