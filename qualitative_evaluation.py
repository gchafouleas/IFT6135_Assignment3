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
parser.add_argument('--load_model', type=str, default='gan/models/7_generator.pt',
                    help='path for loading the model')
parser.add_argument('--image_save_directory', type=str, default='gan/samples/',
                    help='path for loading the model')
parser.add_argument('--num_samples', type=int, default='10',
                    help='number of samples per evaluation')

args = parser.parse_args()

batch_size = 32
model = Generator(batch_size)
model.load_state_dict(torch.load(args.load_model))

#generate samples 
for i in range(args.num_samples):
    #generate image from normal distribution
    noise = Variable(torch.randn(1, 100, 1, 1))
    image = model(noise)
    image = image.detach()
    plt.imshow(image[0].permute(2,1,0))
    plt.savefig(args.image_save_directory +'normal/'+ str(i) + 'image.png', bbox_inches='tight')
    plt.clf()

    #generate image from disturbed normal distribution
    for k in range(25):
        noise[0,k+5,0,0] += 20
    image = model(noise)
    image = image.detach()
    plt.imshow(image[0].permute(2,1,0))
    plt.savefig(args.image_save_directory +'disturbed/'+ str(i) + '_image.png', bbox_inches='tight')
    plt.clf()

#Compare between interpolating in the data space and in the latent space
alpha = np.linspace(0,1,10)
for a in alpha:
    #interpolate latent
    a = round(a,1)
    z_0 = Variable(torch.randn(1, 100, 1, 1))
    z_1 = Variable(torch.randn(1, 100, 1, 1))
    z_a = a*z_0 + (1-a)*z_1
    image = model(z_a)
    image = image.detach()
    plt.imshow(image[0].permute(2,1,0))
    plt.savefig(args.image_save_directory +'interpolate_latent/'+ str(a) + '_image.png', bbox_inches='tight')
    plt.clf()

    #interpolate data 
    x_0 = model(z_0)
    x_1 = model(z_1)
    image = a*x_0 + (1-a)*x_1
    image = image.detach()
    plt.imshow(image[0].permute(2,1,0))
    plt.savefig(args.image_save_directory +'interpolate_data/'+ str(a) + '_image.png', bbox_inches='tight')
    plt.clf()