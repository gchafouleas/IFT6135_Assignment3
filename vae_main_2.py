"""Training for Q2 Vae (MNIST Binary Dataset."""

from torchvision.datasets import utils
import torch.utils.data as data_utils
import torch
import os
import numpy as np
from torch import nn
from torch.nn.modules import upsampling
from torch.functional import F
from torch.optim import Adam
from torch.autograd import Variable
import matplotlib.pyplot as plt

from vae_2 import VAE


DEFAULT_LR = 3e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_EPOCHS = 20

DEFAULT_SAVE_PATH = './q2/'

def train(epoch, model, optimizer, debug=True, save_imgs=True):
    model.train()
    train_loss = 0
    for batch_num, data in enumerate(train_loader):
        data = Variable(data)
        if torch.cuda.is_available():
            data = data.cuda()
        optimizer.zero_grad()
        recon_x, mu, logvar = model(data)
		
        if batch_num == len(train_loader)-1 and debug:
			# Shows original and reconstruction side-by-side
            tst = recon_x.cpu()
            plt.imshow(data.cpu()[0,0])
            plt.show()
            plt.imshow(recon_x.detach().cpu()[0,0])
            plt.show()
		if save_imgs:
			torchvision.utils.save_image(data.data, '{}/images/Train_Epoch_{}_data.jpg'.format(DEFAULT_SAVE_PATH, epoch), nrow=8, padding=2)
			torchvision.utils.save_image(recon_x.data, '{}/images/Train_Epoch_{}_data.jpg'.format(DEFAULT_SAVE_PATH, epoch), nrow=8, padding=2)

        loss = loss_function(recon_batch, data, mu, logvar, loss_fn)
        loss.backward()
		current_loss = loss.data.item()
        train_loss += current_loss
		current_loss /= len(data)
        optimizer.step()
		
        if batch_num % 20 == 0:
			print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, batch_num, current_loss))

			
	# Avergage over num_batches * batch_size
	train_loss /= len(train_loader)*DEFAULT_BATCH_SIZE
            
	print('------ Training Epoch {} Complete. Average loss of {} ------'.format(epoch, train_loss))

    return train_loss
	
	
def valid(epoch, model, save_imgs=True):
    model.eval()
    test_loss = 0
    num = 0
    for batch_num, data in enumerate(valid_loader):
        num+=data.shape[0]
        if torch.cuda.is_available():
            data = data.cuda()
        recon_x, mu, logvar = model(data)
        los = loss_function(recon_x, data, mu, logvar, loss_fn).data.item()
        test_loss += los

		if save_imgs:
			torchvision.utils.save_image(data.data, '{}/images/Valid_Epoch_{}_data.jpg'.format(DEFAULT_SAVE_PATH, epoch), nrow=8, padding=2)
			torchvision.utils.save_image(recon_x.data, '{}/images/Valid_Epoch_{}_data.jpg'.format(DEFAULT_SAVE_PATH, epoch), nrow=8, padding=2)

    test_loss /= (num)
	print('------ Validation Epoch {} Complete. Average loss of {} ------'.format(epoch, test_loss))
    return test_loss
	
	
def test(model, loss_fn, save_imgs=True):
	model.eval()
    test_loss = 0
    num = 0
    for batch_num, data in enumerate(test_loader):
        num+=data.shape[0]
        if torch.cuda.is_available():
            data = data.cuda()
        recon_x, mu, logvar = model(data)
        los = loss_function(recon_x, data, mu, logvar, loss_fn).data.item()
        test_loss += los

		if save_imgs:
			torchvision.utils.save_image(data.data, '{}/images/Test_data.jpg'.format(DEFAULT_SAVE_PATH), nrow=8, padding=2)
			torchvision.utils.save_image(recon_x.data, '{}/images/Valid_data.jpg'.format(DEFAULT_SAVE_PATH), nrow=8, padding=2)

    test_loss /= (num)
	print('------ Test set evaluation complete. Average loss of {} ------'.format(test_loss))
    return test_loss
	

def loss_function(recon_x, x, mu, logvar, loss_fn):
    BCE = loss_fn(recon_x, x)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())    
    ELBO = BCE + KLD
    return ELBO
	
def train_valid_cycle(epochs=DEFAULT_NUM_EPOCHS, load_model=False):

	# Creates VAE Model
	model = VAE()
	
	if load_model:
		model.load_state_dict(torch.load('{}/model19.pt'.format(DEFAULT_SAVE_PATH)))
	
	if torch.cuda.is_available():
		model = model.cuda()
		
	# Creates Optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)
	
	# Defines loss function
	bce_loss = nn.BCELoss(size_average=False)
	
	# Runs Train/Validation loop
	for epoch in range(epochs):
		train(epoch, model, optimizer, loss_fn=bce_loss)
		# Saves the model
		torch.save(model.state_dict(), '{}/model_{}.pt'.format(DEFAULT_SAVE_PATH, epoch))
		valid(epoch, model, loss_fn=bce_loss)
		
	# Runs on Test Set
	test(model, loss_fn=bce_loss)
	
if __name__ == '__main__':

	train_valid_cycle()
