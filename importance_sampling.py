"""Importance sampling for VAE Question #2."""

import scipy as sp
from scipy.stats import norm
from scipy.stats import multivariate_normal
import numpy as np
from torch import nn
from torch.nn.modules import upsampling
from torch.functional import F
from torch.optim import Adam
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.special import logsumexp

DEFAULT_NUM_SAMPLES = 200
DEFAULT_RECON_FUNC = nn.BCELoss(size_average=False)


def generate_z(model, data, num_samples=DEFAULT_NUM_SAMPLES):
    """ Generates samples for z. """
    
    mu, logvar = model.encode(data)
    mu, logvar = mu.detach().cpu().numpy(), logvar.detach().cpu().numpy()
    
    samples = []
    qz = []
    
    for i in range(mu.shape[0]):
        tmp_samples = []
        for j in range(mu.shape[1]):
            tmp_samples.append(
                np.random.normal(
                    mu[i][j], np.exp(0.5 * logvar[i][j]), num_samples
                )
            )
        samples.append(tmp_samples)
            
    samples = np.array(samples)
            
    return samples, mu, logvar
    
def logpz(z):
    """ Computes log(p(z)) for ONE sample.
    
    :param z: The z samples.
    
    """
    
    pz = norm.pdf(z)
    pz = pz.swapaxes(-1,-2)
    
    return np.sum(np.log(pz), axis=-1)
    
def logqzx(z, mu, logvar):
    """ Computes log(q(z|x)) for ONE sample.
    
    :param z: The z samples.
    
    """
    
    qz = []
    
    z_tmp = z.swapaxes(-1, -2)

    for i in range(z_tmp.shape[0]):
        qz.append(norm.pdf(z_tmp[i], loc=mu, scale=np.exp(0.5*logvar)))
    
    qz = np.array(qz)
    
    return np.sum(np.log(qz), axis=-1)

    

def logpxz(x, x_recon, recon_func, num_samples):

    ret_val = np.empty(num_samples)
    
    for sample in range(num_samples):
        bce = recon_func(x_recon[sample], x)
        ret_val[sample] = -bce
        
    return ret_val

def importance_sampling(model, x, z=None, mu=None, logvar=None,
                        generate_samples=True, num_samples=DEFAULT_NUM_SAMPLES,
                        recon_func=DEFAULT_RECON_FUNC, debug=False):
    """Gives a estimate for log(p(x))
    
    M: Batch size.
    D: Unrolled image size.
    L: Latent space size.
    K: Number of samples.
    
    :param model: The trained VAE model.
    :param x: Image array of shape (M,D).
    :param z: Sample array of shape (M,K,L).
    :param mu: Mu used to generate samples.
    :param logvar: Logvariance used to generate samples.
    :param generate_samples: Whether or not to generate samples. If set to True,
        whatever is passed as z,mu and logvar is ignored.
    
    """
    
    if generate_samples:
        if z is not None:
            print('Warning! Generate Samples set to True,'
                  'thus the passed z_i will be ignored')
        
        z, mu, logvar = generate_z(model, x, num_samples)
        
    x = x.reshape(x.shape[0], -1)
    
    M,D = x.shape[0], x.shape[1]
    K,L = z.shape[1], z.shape[2]
    
    if debug:
        print('M:{}, D:{}, K:{}, L:{}'.format(M,D,K,L))
    
    # Computes negative log of number of samples
    n_logk = -np.log(num_samples)
    
    # Loops over the batches
    logpx_estimates = []
    
    z_swapped = torch.from_numpy(z.swapaxes(-1,-2)).cuda().type(torch.float)
    
    for index in range(M):
        
        # Constructs an image from the samples
        x_recon = model.decode(z_swapped[index])
        x_recon = x_recon.reshape(-1, D).detach()
        
        # Exponential sum trick
        logpx = n_logk + logsumexp(
            logpz(z[index]) - logqzx(z[index], mu[index], logvar[index])
            + logpxz(x[index], x_recon, recon_func, num_samples=num_samples)
        )
        
        logpx_estimates.append(logpx)
        
    if debug:
        print(np.mean(np.concatenate(logpx_estimates)))
        
    return np.array(logpx_estimates)
    
if __name__ == '__main__':
    pass
