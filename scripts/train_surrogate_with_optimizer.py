# Author: Kinga Anna Wozniak
# Date: 2025-03-18
# Description: Train surrogate model and optimize mutual information for noisy channel experiments.

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as Toptim
import matplotlib.pyplot as plt
import yaml
from mi_estimopt.ml import surrogate as surr
from mi_estimopt.ml import optimizer as surr_opt
from mi_estimopt.util import data_util as daut


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def plot_theta_vs_mi(theta, mi, scatter_thetas=False, plot_name=None, fig_dir=None):
    # plot theta vs mi
    sorted_indices = np.argsort(theta)
    sorted_theta = theta[sorted_indices]
    sorted_mi = mi[sorted_indices]

    plt.plot(sorted_theta, sorted_mi)
    if scatter_thetas:
        plt.scatter(sorted_theta, sorted_mi, color='red', marker='>')
    plt.xlabel('Theta')
    plt.ylabel('MI')
    plt.title('Theta vs MI')
    if plot_name is not None and fig_dir is not None:
        plt.savefig(f'{fig_dir}/{plot_name}.png')
    plt.show()
    plt.close()


def main():

    #****************************************#
    #    runtime params
    #****************************************#

    config_path = '../config/noisy_channel.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result_dir = '../results/noisy_channel_test/' + 'run_' + str(config['run_n'])


    #****************************************************************#
    #                       set up surrogate model
    #****************************************************************#

    logger.info('setting up surrogate model')

    surr_model = surr.MLP_Surrogate(N_feat=2,device=device)

    metric = nn.MSELoss()
    surr_opt = Toptim.Adam(surr_model.parameters(), lr=0.05)


    #****************************************************************#
    #                       load data
    #****************************************************************#

    # load minf results
    logger.info('loading data from ' + result_dir)
    result_ll = np.load(result_dir+'/result_ll_test.npz')

    # create surrogate dataset
    thetas = np.column_stack((result_ll['theta1'], result_ll['theta2']))
    dataset = daut.make_tensor_dataset(thetas, result_ll['mi'])


    plot_theta_vs_mi(result_ll['theta1'], result_ll['mi'], plot_name='theta1_vs_mi', fig_dir=result_dir)
    plot_theta_vs_mi(result_ll['theta2'], result_ll['mi'], plot_name='theta2_vs_mi', fig_dir=result_dir)

    # define data loaders
    train_split = 0.95
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_split, 1-train_split])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    #****************************************************************#
    #                       train surrogate model

    surr_opt = Toptim.Adam(surr_model.parameters(), lr=config['lr_surr'])

    scheduler = Toptim.lr_scheduler.StepLR(surr_opt, step_size=20, gamma=0.1)

    logger.info('training surrogate model')

    surr_loss_train, surr_loss_valid = surr.train_surrogate(surr_model=surr_model, surr_metric=metric, surr_opt=surr_opt, \
                                                            scheduler=scheduler, dataloader=train_loader, data_valid=val_loader, n_epochs=config['n_epochs_surr'])


    #****************************************************************#
    #                       find theta with best MI
    #****************************************************************#

    surr_optimizer = surr_opt.Optimizer(surr_dataset=dataset, surrogate=surr_model, epoch_n=300)
    thetas = surr_optimizer.optimize()

    # get mutual information for thetas

    thetas = torch.tensor(thetas)
    mis = surr_model(thetas)

    thetas_np = thetas.detach().numpy().squeeze()
    mis_np = mis.detach().numpy().squeeze()

    plot_theta_vs_mi(thetas_np[:,0], mis_np, scatter_thetas=True, plot_name='theta1_vs_mi_descent', fig_dir=result_dir)
    plot_theta_vs_mi(thetas_np[:,1], mis_np, scatter_thetas=True, plot_name='theta2_vs_mi_descent', fig_dir=result_dir)