# ########################################################################################
# File: src/mi_estimopt/ml/mine.py
# Date: 2025-03-18
# Mutual Information Estimation using MINE (Mutual Information Neural Estimation)
# Based on the original MINE paper: https://arxiv.org/abs/1801.04062
# and on the implementation by Francois Fleuret
# Author: Kinga Anna Wozniak
# ########################################################################################

import math
import sys
import torch
from torch import nn
import torch.nn.functional as F


##################################
#               model
##################################

acti_dd = { 'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'elu': nn.ELU() , 'leaky': nn.LeakyReLU() }


class MI_Model(nn.Module):

    def __init__(self, B_N:int, encoder_N:int=128, hidden_N:int=32, acti:str='relu', acti_out:str=None, device:str='cpu'):

        super(MI_Model, self).__init__()

        self.device = torch.device(device)

        # encoder for variable of interest / target (e.g. true energy)
        self.features_a = nn.Sequential(
            nn.Linear(1, hidden_N), acti_dd[acti],
            nn.Linear(hidden_N, hidden_N), acti_dd[acti],
            nn.Linear(hidden_N, encoder_N), acti_dd[acti],
        )

        # encoder for informing variables (e.g. measured energy)
        self.features_b = nn.Sequential(
            nn.Linear(B_N, hidden_N), acti_dd[acti],
            nn.Linear(hidden_N, hidden_N), acti_dd[acti],
            nn.Linear(hidden_N, encoder_N), acti_dd[acti],
        )

        connected_mlp = []
        connected_mlp.append(nn.Linear(encoder_N*2, 200))
        connected_mlp.append(acti_dd[acti])
        connected_mlp.append(nn.Linear(200, 1))
        if acti_out is not None:
            connected_mlp.append(acti_dd[acti_out])

        self.fully_connected = nn.Sequential(*connected_mlp)

    def forward(self, a, b):
        a = self.features_a(a).view(a.size(0), -1)
        b = self.features_b(b).view(b.size(0), -1)
        x = torch.cat((a, b), 1) # first dimension is batch-dimension
        return self.fully_connected(x)


def mutual_info(dep_ab:torch.Tensor, indep_ab:torch.Tensor, eps:float=1e-8):

    return dep_ab.mean() - torch.log(indep_ab.exp().mean()+eps) # means over batch


def train(model: MI_Model, dataloader, nb_epochs, optimizer, eps=1e-8) -> float:

    model.train()

    train_mi = []

    for e in range(nb_epochs):
        
        acc_mi = 0.0
        
        for batch in dataloader:
            
            batch_a, batch_b, batch_br, _ = [b.to(model.device) for b in batch]

            # apply the model: pass a & b and a & b_permuted
            dep_ab = model(batch_a, batch_b)
            indep_ab = model(batch_a, batch_br)

            mi = mutual_info(dep_ab=dep_ab, indep_ab=indep_ab, eps=eps)
            loss = -mi
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            acc_mi += mi.item()
        
        acc_mi /= len(dataloader)  # mi per batch
        acc_mi /= math.log(2)
        
        train_mi.append(acc_mi)
        print(f'{e+1} {acc_mi:.04f}\n')

    return acc_mi


def test(model, dataloader, eps:float=1e-8) -> float:

    model.eval()
    test_acc_mi = 0.0

    for batch in dataloader:

        batch_a, batch_b, batch_br, _ = [b.to(model.device) for b in batch]

        dep_ab = model(batch_a, batch_b)
        indep_ab = model(batch_a, batch_br)
        
        mi = mutual_info(dep_ab=dep_ab, indep_ab=indep_ab, eps=eps)
        test_acc_mi += mi.item()

    test_acc_mi /= len(dataloader)
    test_acc_mi /= math.log(2)

    return test_acc_mi
