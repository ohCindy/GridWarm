import os
from os import getcwd
from sys import path

import os

main_dir = getcwd() # os.path.dirname(getcwd()) #colab
path.append(main_dir + "/ML")

import ML_global_vars

HIDDEN_DIM = ML_global_vars.HIDDEN_DIM  # width of NN 32 doesn't work well for 'dV' mode
N_HIDDEN = ML_global_vars.N_HIDDEN

from nn_extensions import weight_initializer
from nn_extensions import build_NN

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F  # activation functions
import time

class Model_cGRF(nn.Module):
    def __init__(self, c_type='MadIoT', method_name='cGRF',
                 apply_parameter_sharing=True, BusNumList=[],
                 apply_ZI=False,
                 hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN,
                 apply_feature_mapping=False):
        """
        num_tags: how many output each node has  2: real and imaginary
        c_type: type of contingency 'MadIoT','substation outage','gen outage','line outage'
        """
        super(Model_cGRF, self).__init__()
        self.method_name = method_name
        num_tags = 2  # [vr,vimag]
        self.num_tags = num_tags
        self.apply_feature_mapping = apply_feature_mapping
        self.apply_parameter_sharing = apply_parameter_sharing  # if true, use one NN-node/edge for all nodes/edges; else each node/edge has its NN
        self.BusNumList = BusNumList
        self.apply_ZI = apply_ZI  # domain knowledge of ZI bus: eta=0 at ZI buses
        #
        self.c_type = c_type
        if c_type == 'line outage':
            node_input_dim = 7  # Vr, Vi, Pinj Qinj, Irinj, Iiinj, Qshunt precontingency
            edge_input_dim = 3  # G,B,Bsh
        elif c_type == 'gen outage':
            node_input_dim = 8  # +dPgen
            edge_input_dim = 3
        else:
            node_input_dim = 10  # +dPgen dPload dQload
            edge_input_dim = 3
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        #
        node_feature_dim = node_input_dim * 2 + edge_input_dim + 1  # feature=[node input, neighbor node, neighbor edge, distance]
        edge_feature_dim = edge_input_dim + node_input_dim * 2 + 1  # feature[edge input, from node input, to node, distance]
        """
        if apply_feature_mapping:
            # feature mapping using polynomial mapping: [x1,x2]=>[x1^2,x1y1,y2^2] d-> d+d(d+1)/2
            node_feature_dim = (node_feature_dim - 1) ** 2 + 1
            edge_feature_dim = (edge_feature_dim - 1) ** 2 + 1
        print("node feature dim: ", node_feature_dim, "edge feature dim: ", edge_feature_dim)
        """
        #
        if self.apply_parameter_sharing:
            self.NN_node = build_NN(node_feature_dim, num_tags + num_tags * num_tags, hidden_dim, n_hidden)
            self.NN_edge = build_NN(edge_feature_dim, num_tags ** 2, hidden_dim, n_hidden)
        else:
            self.NN_nodes_book = dict()
            for i, busnum in enumerate(BusNumList):
                self.NN_nodes_book[busnum] = i
            NN_nodes = [build_NN(node_feature_dim, num_tags + num_tags * num_tags, hidden_dim, n_hidden) for busnum in
                        BusNumList]
            self.NN_nodes = nn.ModuleList(NN_nodes)  # a list of NNs, each is one NN-node
            self.NN_edge = build_NN(edge_feature_dim, num_tags ** 2, hidden_dim, n_hidden)
            print("%d NN-nodes defined" % len(NN_nodes))
        # Initalize weights
        self.init_weights()

    def init_weights(self, case_name=None, epoch2load=None, method_name=None):
        if not method_name:
            method_name = self.method_name
        if epoch2load:
            self.load(case_name, epoch2load, method_name)  # self.load('case30',49)
            print('load from existing model', case_name, epoch2load, method_name)
        else:
            if self.apply_parameter_sharing:
                self.NN_node.apply(weight_initializer)
                self.NN_edge.apply(weight_initializer)
            else:
                self.NN_nodes.apply(weight_initializer)
                self.NN_edge.apply(weight_initializer)
            print("NN weights initialized by xavier")


    def forward(self, feat_i, feat_ni, feat_ne, feat_di,
                feat_e, feat_st, feat_de,
                bus_nums, if_ZIs, edge_st_nums):
        """
        x is the input data of one contingency sample, taking build_grid() outputs
        Distances is the pre-calculated distances telling the distance of each node and edge from the contingency location
        if Distances is empty, we use function get_node_distance_from_contingency() to calcualte
        feat_i: torch tensor (n_activebus, n_raw_node_features), raw node features
        feat_ni: torch tensor (n_activebus, n_raw_node_features), sum of neighboring node features
        feat_ne: torch tensor (n_activebus, n_raw_edge_features), sum of features at neighbor edges of node i
        feat_di: torch tensor (n_activebus,), distance(node,contingency)
        feat_e: torch tensor (n_activeedge,n_raw_edge_features), raw edge features
        feat_st: torch tensor (n_activeedge,2*n_raw_node_features), feature at end nodes of edge
        feat_de: torch tensor (n_activeedge,), distance(edge, contingency)
        bus_nums: (n_activebus) a torch vector of active bus numbers
        if_ZIs: (n_activebus) a torch vector of "whether or not each active bus is ZI",
        edge_st_nums: torch tensor (n_activebbus, 2), edge_st_nums[e] = [from_bus_num, to_bus_num]
        #
        input to NN-node: concatenate (feat_i, feat_ni, feat)
        """
        # truncate inputs
        feat_i = feat_i[:,0:self.node_input_dim]
        feat_ni = feat_ni[:,0:self.node_input_dim]
        feat_ne = feat_ne[:,0:self.edge_input_dim]
        feat_e = feat_e[:,0:self.edge_input_dim]
        feat_st = feat_st[:,0:2*self.node_input_dim]
        feat_di = feat_di.view(-1,1)
        feat_de = feat_de.view(-1,1)
        #
        nn_node_in = torch.cat((feat_i,feat_ni,feat_ne,feat_di),dim=1)
        if self.apply_parameter_sharing:
            nn_node_out = self.NN_node(nn_node_in)
        else:
            nn_node_out = torch.stack([self.NN_nodes[self.NN_nodes_book[int(Bus)]](nn_node_in[i,:])
            for i,Bus in enumerate(bus_nums)])
        eta_i_vec = nn_node_out[:,0:self.num_tags]
        Lambda_i_vec = nn_node_out[:,self.num_tags:]
        nn_edge_in = torch.cat((feat_e,feat_st,feat_de),dim=1)
        Lambda_e_vec = self.NN_edge(nn_edge_in)
        # domain knowledge of ZI bus:
        if self.apply_ZI:
            eta_i_vec[if_ZIs==True] = 0*eta_i_vec[if_ZIs==True]
        # 4. pack results into sparse Lambda and eta matrix:
        # Lambda should be a sparse coo matrix of size 2N by 2N, eta should be a dense vector of size 2N by 1
        #   1) make dense vector eta:
        out_eta = eta_i_vec.contiguous().view(-1,1)
        #   2) make sparse matrix lambda:
        n_active_nodes = len(bus_nums)
        rows = np.array([np.array([0,0,1,1])+2*i for i in range(n_active_nodes)])  # lambda ii
        cols = np.array(
            [np.array([0,1,0,1])+2*i for i in range(n_active_nodes)])
        rows = np.concatenate((rows,
                               np.array([np.array([0,0,1,1])+2*(bus_nums==from_bus).nonzero().item()
                               for from_bus,to_bus in edge_st_nums])),
                              axis=0)
        cols = np.concatenate((cols,
                               np.array([np.array([0,1,0,1])+2*(bus_nums==to_bus).nonzero().item()
                             for from_bus,to_bus in edge_st_nums])), axis=0)
        rows = rows.flatten().tolist()
        cols = cols.flatten().tolist()
        #
        sz = (2*n_active_nodes, 2*n_active_nodes)  # size of Lambda matrix: 2N by 2N
        Lambda = torch.sparse_coo_tensor([rows, cols],
                                         torch.cat((Lambda_i_vec.contiguous().view(-1),
                                                    Lambda_e_vec.contiguous().view(-1))),
                                         sz).coalesce()
        out_Lambda = (Lambda.transpose(0,
                                       1).coalesce() + Lambda) / 2  # make Lambda symmetric (since it's the inverse of covariance)
        return out_Lambda, out_eta
        # Lambda is torch.sparse_coo_tensor 2N by 2N, out_eta is dense tensor of size 2N by 1

    def save(self, case_name, epoch):
        output_dir = getcwd() + '/ML_Output/%s/' % (case_name+self.c_type)
        output_file_name = output_dir + case_name + self.method_name + '_epoch%d' % epoch + '.pth'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.state_dict(), output_file_name)
        print("model has been saved at ", output_file_name)

    def load(self, case_name,  epoch=49, method_name=None):
        if not method_name:
            method_name = self.method_name
        output_file_name = getcwd() + '/ML_Output/%s/' % (case_name+self.c_type) + case_name + method_name + '_epoch%d' % epoch + '.pth'
        print("loading ", output_file_name)
        self.load_state_dict(torch.load(output_file_name))
        print("model loaded from ", output_file_name)

    def copy_PS_model(self, model_ps):
        #init non-PS model parameter with a PS model
        if (not self.apply_parameter_sharing) and (model_ps.apply_parameter_sharing):
            self.NN_edge.load_state_dict(model_ps.NN_edge.state_dict())
            for ib in range(len(self.NN_nodes)):
                self.NN_nodes[ib].load_state_dict(model_ps.NN_node.state_dict())
            print('parameters copied from a PS model')

