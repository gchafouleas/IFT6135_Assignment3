#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


from __future__ import print_function
import numpy as np
import torch 
import matplotlib.pyplot as plt
import samplers as samplers
from discriminator import Discriminator
import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch model')
parser.add_argument('--loss_type', type=str, default='JSD',
                    help='loss to use; JSD WD')
parser.add_argument('--question', type=int, default='3',
                    help='loss to use; 3 kor 4')


args = parser.parse_args()

# plot p0 and p1
plt.figure()
torch.manual_seed(1111)
# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)
# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))
plt.clf()

############### import the sampler ``samplers.distribution4'' 
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######
directory = "model/"
num_epochs = 1000 

if args.question ==3:
    print("question 3")
    phi = np.linspace(-1,1, 21)
    x = samplers.distribution1(0)
    values = []
    for i in phi : 
        y = samplers.distribution1(i)
        model = Discriminator(2, 50, 512, 0) 
        for epoch in range(num_epochs):
            x_batch = torch.from_numpy(next(x))
            y_batch = torch.from_numpy(next(y))
            model.train(x_batch.type(torch.FloatTensor),y_batch.type(torch.FloatTensor), args.loss_type)
        #torch.save(model.state_dict(), os.path.join(directory, 'best_params_'+str(i)+'.pt'))
        x_dist = samplers.distribution1(0,10000)
        y_dist = samplers.distribution1(i,10000)
        x_dist_batch = torch.from_numpy(next(x_dist))
        y_dist_batch = torch.from_numpy(next(y_dist)) 
        x_value = x_dist_batch.type(torch.FloatTensor)
        y_value = y_dist_batch.type(torch.FloatTensor)
        if args.loss_type == "JSD":
            print("JSD")
            jsd = model.loss_JSD(model.forward(x_dist_batch.type(torch.FloatTensor)), model.forward(y_dist_batch.type(torch.FloatTensor)))
            values.append(-jsd)
        elif args.loss_type == "WD":
            wd = torch.mean(model.forward(x_value) - model.forward(y_value))
            values.append(wd)

    plt.plot(phi,values, 'o-')
    if args.loss_type == "JSD":
        plt.ylabel("JSD")
        plt.xlabel("phi")
        plt.title("JSD vs phi")
        plt.savefig(directory + '_JSD_phi.png', bbox_inches='tight')
    elif args.loss_type == "WD":
        plt.ylabel("Wasserstein Distance")
        plt.xlabel("Phi")
        plt.title("WD vs. Phi")
        plt.savefig(directory + '_WD_phi.png', bbox_inches='tight')
    plt.clf()    

if args.question == 4:
    f1_dist = samplers.distribution3(512)
    f0_dist = samplers.distribution4(512)
    model = Discriminator(1, 50, 512, 0)
    for epoch in range(num_epochs):
        f1_batch = torch.from_numpy(next(f1_dist))
        f0_batch = torch.from_numpy(next(f0_dist))
        model.train(f1_batch.type(torch.FloatTensor),f0_batch.type(torch.FloatTensor))
    f1_value = []
    discriminator_outputs = []
    for batch in xx: 
        f0 = samplers.distribution3(np.abs(int(batch)))
        x = next(f0)
        f0_batch = torch.from_numpy(next(f0))
        x = f0_batch.type(torch.FloatTensor)
        output = model(x)
        f1 = (x * output)/(1 - output)
        f1_value.append(f1)
        discriminator_outputs.append(output)

    plt.plot(f1_value, 'o-')
    plt.plot(discriminator_outputs, 'o-')
    plt.ylabel("y")
    plt.xlabel("x")
    plt.title("Discriminator output and estimated f1")
    plt.legend(labels = ["F1", "D(x)"])
    plt.savefig(directory + '_question_4.png', bbox_inches='tight')

















############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density



r = xx # evaluate xx using your discriminator; replace xx with the output
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r)
plt.title(r'$D(x)$')

estimate = np.ones_like(xx)*0.2 # estimate the density of distribution4 (on xx) using the discriminator; 
                                # replace "np.ones_like(xx)*0." with your estimate
plt.subplot(1,2,2)
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')











