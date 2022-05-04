
import numpy as np
import torch
import torch.nn as nn

#https://github.com/pytorch/pytorch/issues/36459
#https://stackoverflow.com/questions/60586559/parallel-analog-to-torch-nn-sequential-container
#parallel modules share input

def weight_initializer(param):
    if type(param) == nn.Linear:
        torch.nn.init.xavier_uniform_(param.weight)
        #torch.nn.init.xavier_normal_(param.weight)
        param.bias.data.fill_(0.0)


def build_NN(in_dim,out_dim,hidden_dim,n_hidden):
    layers = []
    # input
    layers.append(nn.Linear(in_dim, hidden_dim))
    layers.append(nn.LeakyReLU())
    # hidden layers
    for l in range(n_hidden):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.LeakyReLU())
    # output layer
    layers.append(nn.Linear(hidden_dim, out_dim))
    #
    NN = nn.Sequential(*layers)
    return NN

