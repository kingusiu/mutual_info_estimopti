######################################################################################### 
# This code is part of the mutual information estimation project.
# It is used to train and test a mutual information model on a noisy channel dataset.
# The script generates samples, trains the model, and evaluates its performance.
# Author: Kinga Anna Wozniak
# Date: 2023-10-30
########################################################################################

import os
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn import feature_selection
import yaml
import random
import logging
import matplotlib.pyplot as plt

# Add src to sys.path for development/demo purposes
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mi_estimopt.dats import input_generator as inge
from mi_estimopt.ml import mine



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

corrs_ll = lambda tmin,tmax,tstep: {f'corr {corr:.03f}': round(corr, 3) for corr in np.arange(tmin, tmax, tstep)}

def plot_inputs(A_list, B_list, t1_list, t2_list, plot_name='scatter_plot', fig_dir='results'):

    num_rows_cols = int(np.sqrt(len(t1_list)))
    fig, axs = plt.subplots(num_rows_cols, num_rows_cols, figsize=(6*len(t1_list), 8*len(t2_list)))
    
    for i in range(num_rows_cols):
        for j in range(num_rows_cols):
            idx = i * num_rows_cols + j
            axs[i, j].scatter(A_list[idx], B_list[idx])
            axs[i, j].set_xlabel('A')
            axs[i, j].set_ylabel('B')
            axs[i, j].set_title(f'theta1={t1_list[idx]:.03f}, theta2={t2_list[idx]:.03f}')
    
    plt.show()


def plot_histogram(thetas, thetas_nominal, plot_name='theta_histogram', fig_dir='results'):
    num_plots = len(thetas)
    fig, axs = plt.subplots(1, num_plots, figsize=(6*num_plots,8))
    for i, (theta, theta_nominal) in enumerate(zip(thetas, thetas_nominal)):
        axs[i].hist(theta, bins=50, alpha=0.5, label='theta distribution')
        axs[i].axvline(x=theta_nominal, color='red', linestyle='--', label=f'theta={theta_nominal:.03f}')
        axs[i].set_xlabel('Theta')
        axs[i].set_ylabel('Frequency')
        axs[i].legend()
    plt.show()


def plot_results(result_ll, plot_name='mi_vs_theta', fig_dir='results'):

    result_ll = np.array(result_ll)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = result_ll[:, 0]
    y = result_ll[:, 1]
    z = result_ll[:, 2]

    ax.plot_trisurf(x, y, z, cmap='viridis')

    ax.set_xlabel('theta1: noise')
    ax.set_ylabel('theta2: damp')
    ax.set_zlabel('approx MI')

    plt.show()


def make_two_theta_grid(theta_min, theta_max, theta_num):
    t1 = np.linspace(theta_min, theta_max, theta_num)
    t2 = np.linspace(1, theta_max, theta_num)
    random.shuffle(t1)
    random.shuffle(t2)
    tt1,tt2 = np.meshgrid(t1, t2)
    return tt1, tt2


def make_tensor_dataset(*np_arrays):
    '''
    Convert numpy arrays to torch tensors and save to torch dataset.
    '''
    tensors = [torch.from_numpy(arr) for arr in np_arrays]
    tensors = [t.unsqueeze(1) if t.ndim == 1 else t for t in tensors]
    return torch.utils.data.TensorDataset(*tensors)


def main():

    #****************************************#
    #    runtime params
    #****************************************#

    config_path = '../config/noisy_channel.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    #****************************************#
    #               build model 
    #****************************************#

    B_N = 1

    # create model
    model = mine.MI_Model(B_N=B_N, acti=config['activation'], acti_out=config['activation_out'])
    model.to(rtut.device)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    #****************************************#
    #               load data 
    #****************************************#
    N_per_theta = config['n_per_theta']

    tt1, tt2 = make_two_theta_grid(config['theta_min'],config['theta_max'],config['theta_step'])

    result_ll = []

    data_dict = {'A_train': [], 'B_train': [], 'tt1_train': [], 'tt2_train': []}

    for t1, t2 in zip(tt1.flatten(), tt2.flatten()):
        
        logger.info(f'generating data for t1: {t1:.03f}, t2: {t2:.03f}')

        A_train, B_train, thetas_train, *_ = inge.generate_two_theta_noisy_samples(N=N_per_theta, t1_noise_nominal=t1, t2_damp_nominal=t2)

        data_dict['A_train'].append(A_train)
        data_dict['B_train'].append(B_train)
        data_dict['tt1_train'].append(thetas_train[:,0])
        data_dict['tt2_train'].append(thetas_train[:,1])

        dataset_train = make_tensor_dataset(A_train, B_train, thetas_train)
        train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)

        #****************************************#
        #               train model
        #****************************************#
        train_acc_mi = mine.train(model, train_dataloader, config['n_epochs'], optimizer)
        train_true_mi = feature_selection.mutual_info_regression(A_train.reshape(-1,1), B_train)[0]

        #****************************************#
        #              print results    
        # ****************************************#
        logger.info(f'theta1 {t1:.03f} / theta2 {t2:.03f}: \t train MI {train_acc_mi:.04f} \t true MI {train_true_mi:.04f}')    
        result_ll.append([t1, t2, train_acc_mi, train_true_mi])
    
    plot_inputs(data_dict['A_train'], data_dict['B_train'], tt1.flatten(), tt2.flatten(), plot_name='scatter_plot_inputs_train')
    
    xlabel = 'Theta/noise level' if 'noise' in config['theta_type'] else 'Theta/correlation'
    plot_results(result_ll, plot_name='mi_vs_theta_train')
    plot_histogram(data_dict['tt1_train'], tt1.flatten(), plot_name='t1_train_histogram')
    plot_histogram(data_dict['tt1_train'], tt2.flatten(), plot_name='t2_train_histogram')



    N_per_theta = config['n_per_theta']
    result_ll = []
    tt1_test, tt2_test = make_two_theta_grid(config['theta_min'], config['theta_max'], config['theta_step'])
    data_dict = {'A_test': [], 'B_test': [], 'tt1_test': [], 'tt2_test': []}
    for t1, t2 in zip(tt1_test.flatten(), tt2_test.flatten()):
        logger.info(f'generating data for t1: {t1:.03f}, t2: {t2:.03f}')
        A_test, B_test, thetas_test, *_ = inge.generate_two_theta_noisy_samples(N=N_per_theta, t1_noise_nominal=t1, t2_damp_nominal=t2)
        data_dict['A_test'].append(A_test)
        data_dict['B_test'].append(B_test)
        data_dict['tt1_test'].append(thetas_test[:, 0])
        data_dict['tt2_test'].append(thetas_test[:, 1])

        dataset_test = make_tensor_dataset(A_test, B_test, thetas_test)
        test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=False)
        #****************************************#
        #               test model
        #****************************************#
        test_acc_mi = mine.test(model, test_dataloader)
        test_true_mi = feature_selection.mutual_info_regression(A_test.reshape(-1, 1), B_test)[0]
        #****************************************#
        #              print results    
        # ****************************************#
        logger.info(f'theta1 {t1:.03f} / theta2 {t2:.03f}: \t test MI {test_acc_mi:.04f} \t true MI {test_true_mi:.04f}')    
        result_ll.append([t1, t2, test_acc_mi, test_true_mi])

    plot_inputs(data_dict['A_test'], data_dict['B_test'], tt1_test.flatten(), tt2_test.flatten(), plot_name='scatter_plot_inputs_test')
    plot_results(result_ll, plot_name='mi_vs_theta_test')
    plot_histogram(data_dict['tt1_test'], tt1_test.flatten(), plot_name='t1_test_histogram')
    plot_histogram(data_dict['tt2_test'], tt2_test.flatten(), plot_name='t2_test_histogram')


if __name__ == "__main__":
    main()