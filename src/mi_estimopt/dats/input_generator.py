import math
import torch
import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import multivariate_normal

from typing import Union



def calc_train_test_split_N(N,train_test_split_share):
    if train_test_split_share:
        return int(N*train_test_split_share)
    return None


################################################
#       generate random correlated variables
# x ... true energy
# y ... deposited energy
###############################################

def generate_random_variables(corr: float = 0., N: int = int(1e5), means: list = None, stds: list = None, train_test_split= None):

    """
    Generate two correlated random variables with a specified correlation coefficient.
    Parameters:
        corr (float): Correlation coefficient between the two variables.
        N (int): Total number of samples to generate.
        means (list): Means of the two variables. Defaults to [0, 0].
        stds (list): Standard deviations of the two variables. Defaults to [1, 1].
        train_test_split (float or None): If float, indicates the proportion of data to use for training.
            If None, the entire dataset is used for training.
    Returns:
        Tuple of numpy arrays: (train inputs, train outputs, test inputs, test outputs).
    """ 

    means = [0, 0] if means is None else means
    stds = [1, 1] if stds is None else stds 

    train_test_split = calc_train_test_split_N(N,train_test_split)

    cov = [[stds[0]**2, stds[0]*stds[1]*corr], [stds[0]*stds[1]*corr, stds[1]**2]]
    normal = multivariate_normal(means, cov, allow_singular=True) 
    A, B = normal.rvs(size=N).astype(np.float32).T

    return A[:train_test_split], B[:train_test_split], A[train_test_split:], B[train_test_split:]



###############################################################
#       generate random bi-modal gaussian mixture variables
# x ... true energy
# y ... deposited energy
##############################################################

def samples_from_multivariate_multimodal_gaussian(mus: Union[list,np.ndarray], covs: Union[list,np.ndarray], N_samples: int = 100) -> np.ndarray:

    # set up distribution

    N_dims = len(mus[0])
    N_modes = len(mus)

    mixtures = [scipy.stats.multivariate_normal(mus[i], covs[i]) for i in range(N_modes)]

    # generate samples

    pick_mode = np.random.choice(N_modes, N_samples)
    N_samples_per_mode = [sum(pick_mode == i) for i in range(N_modes)]

    samples_per_mode = [mixtures[i].rvs(N_samples_per_mode[i]) for i in range(N_modes)]
    samples = np.concatenate(samples_per_mode)
    np.random.shuffle(samples)

    return samples


def generate_bimodal_gauss_mixture_samples(mus, N=int(1e5), train_test_split=None):

    ''' Generate samples for a bimodal Gaussian mixture testcase.
    Parameters:
        mus (list or np.ndarray): List of means for the two modes.
        N (int): Total number of samples to generate.
        train_test_split (float or None): If float, indicates the proportion of data to use for training.
            If None, the entire dataset is used for training.
    Returns:
        Tuple of numpy arrays: (train inputs, train outputs, test inputs, test outputs).
    '''

    train_test_split = calc_train_test_split_N(N,train_test_split)

    covs = [np.eye(2)]*3
    samples = samples_from_multivariate_multimodal_gaussian(mus, covs, N)
    A, B = samples[:,0].astype(np.float32), samples[:,1].astype(np.float32)
    A = torch.from_numpy(A).unsqueeze(-1)
    B = torch.from_numpy(B).unsqueeze(-1)

    return A[:train_test_split], B[:train_test_split], A[train_test_split:], B[train_test_split:]


###############################################################
#       generate noisy channel variables
# x ... true signal sent
# y ... signal received
##############################################################

def generate_noisy_channel_samples(N=int(1e5), noise_std_nominal=0.1, train_test_split=None):

    '''
    Generate samples for a noisy channel testcase with a single parameter.
    The two random variables are some true signal and a noisy measurement of it.
    The noise is parameterized by a single parameter: the noise standard deviation.

    Parameters:
        N (int): Total number of samples to generate.
        noise_std_nominal (float): Nominal value for the noise standard deviation.
        train_test_split (float or None): If float, indicates the proportion of data to use for training.
            If None, the entire dataset is used for training.
    Returns:
        Tuple of numpy arrays: (train inputs, train outputs, train thetas, test inputs, test outputs, test thetas).
    '''

    idx = calc_train_test_split_N(N,train_test_split)

    in_sig = np.random.uniform(0, 4, N).astype(np.float32)
    noise_std = np.abs(np.random.normal(noise_std_nominal,0.05,N)).astype(np.float32)
    noise = np.random.normal(0, noise_std, N).astype(np.float32)
    out_sig = in_sig + noise

    # return x,y,noise(=theta) for train and test
    return in_sig[:idx], out_sig[:idx], noise_std[:idx], in_sig[idx:], out_sig[idx:], noise_std[idx:]


def generate_two_param_noisy_samples(N=int(1e5), t1_noise_nominal=0.1, t2_damp_nominal=1.1, train_test_split=None):

    '''
    Generate samples for a two-parameter noisy channel testcase.
    The two random variables are some true signal and a noisy measurement of it.
    The noise is parameterized by two parameters: the noise standard deviation and a damping factor.

    Parameters:
        N (int): Total number of samples to generate.
        t1_noise_nominal (float): Nominal value for the noise standard deviation.
        t2_damp_nominal (float): Nominal value for the damping factor.
        train_test_split (float or None): If float, indicates the proportion of data to use for training.
            If None, the entire dataset is used for training.
    Returns:
        Tuple of numpy arrays: (train inputs, train outputs, train thetas, test inputs, test outputs, test thetas).
    '''

    idx = calc_train_test_split_N(N,train_test_split)

    in_sig = np.linspace(0, 1, N).astype(np.float32)*4.0

    noise_std = np.abs(np.random.normal(t1_noise_nominal,0.05,N)).astype(np.float32)
    noise = np.random.normal(0, noise_std, N).astype(np.float32)
    
    damp = np.abs(np.random.normal(t2_damp_nominal,0.05,N)).astype(np.float32)

    out_sig = in_sig + noise - np.log(damp)*in_sig
    thetas = np.concatenate((noise_std.reshape(-1, 1), damp.reshape(-1, 1)), axis=1)

    return in_sig[:idx], out_sig[:idx], thetas[:idx], in_sig[idx:], out_sig[idx:], thetas[idx:]
