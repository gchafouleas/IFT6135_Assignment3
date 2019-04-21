import torch 
import torch.nn as nn
import torch.optim as optim 

import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from Gan_model import Discriminator
from Gan_model import Generator as Generator_GAN
from generator import Generator as Generator_VAE
import classify_svhn as data
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import os
import argparse
import torchvision

parser = argparse.ArgumentParser(description='PyTorch model')
parser.add_argument('--load_model', type=str, default='vae/models/4_decoder.pt',
                    help='path for loading the model')
parser.add_argument('--image_save_directory', type=str, default='vae/samples/',
                    help='path for loading the model')
parser.add_argument('--num_samples', type=int, default='10',
                    help='number of samples per evaluation')
parser.add_argument('--model_type', type=str, default='VAE',
                    help='model type: GAN or VAE')

args = parser.parse_args()

if not os.path.exists(args.image_save_directory):
    print("creating qualitative evaluation directory: ", args.image_save_directory)
    os.mkdir(args.image_save_directory)
    os.mkdir(args.image_save_directory + '/normal')
    os.mkdir(args.image_save_directory + '/disturbed')
    os.mkdir(args.image_save_directory + '/interpolate_latent')
    os.mkdir(args.image_save_directory + '/interpolate_data')

batch_size = 32
if args.model_type == "GAN":
    model = Generator_GAN(batch_size)
elif args.model_type == "VAE":
    model = generator = Generator_VAE(latent_size=100)

model.load_state_dict(torch.load(args.load_model))

#generate samples 
for i in range(args.num_samples):
    #generate image from normal distribution
    noise = Variable(torch.randn(32, 100))
    image = model(noise)
    torchvision.utils.save_image(image, args.image_save_directory +'normal/'+ str(i) + 'image.png', nrow=8, padding=2)

    #generate image from disturbed normal distribution
    for k in range(25):
        noise[0,k+5] += 20
    image = model(noise)
    torchvision.utils.save_image(image, args.image_save_directory +'disturbed/'+ str(i) + 'image.png', nrow=8, padding=2)


#Compare between interpolating in the data space and in the latent space
alpha = np.linspace(0,1,10)
for a in alpha:
    #interpolate latent
    a = round(a,1)
    z_0 = Variable(torch.randn(args.num_samples, 100))
    z_1 = Variable(torch.randn(args.num_samples, 100))
    z_a = a*z_0 + (1-a)*z_1
    image = model(z_a)
    torchvision.utils.save_image(image, args.image_save_directory +'interpolate_latent/'+ str(a) + 'image.png', nrow=8, padding=2)

    #interpolate data 
    x_0 = model(z_0)
    x_1 = model(z_1)
    image = a*x_0 + (1-a)*x_1
    torchvision.utils.save_image(image, args.image_save_directory +'interpolate_data/'+ str(a) + 'image.png', nrow=8, padding=2)
