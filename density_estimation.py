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

phi = np.linspace(-1,1, 21)

num_epochs = 1000 
x = samplers.distribution1(0)
values = []
for i in phi : 
    y = samplers.distribution1(i)
    model = Discriminator(2, 50, 512, 0) 
    for epoch in range(num_epochs):
        x_batch = torch.from_numpy(next(x))
        y_batch = torch.from_numpy(next(y))
        model.train(x_batch.type(torch.FloatTensor),y_batch.type(torch.FloatTensor), "WD")
    #torch.save(model.state_dict(), os.path.join(directory, 'best_params_'+str(i)+'.pt'))
    x_dist = samplers.distribution1(0,10000)
    y_dist = samplers.distribution1(i,10000)
    x_dist_batch = torch.from_numpy(next(x_dist))
    y_dist_batch = torch.from_numpy(next(y_dist)) 
    x_value = x_dist_batch.type(torch.FloatTensor)
    y_value = y_dist_batch.type(torch.FloatTensor)
    wd = torch.mean(model.forward(x_value) - model.forward(y_value))
    values.append(wd)


plt.plot(phi,values, 'o-')
plt.ylabel("Wasserstein Distance")
plt.xlabel("Phi")
plt.title("WD vs. Phi")
plt.savefig(directory + '_WD_phi.png', bbox_inches='tight')
plt.clf()















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











