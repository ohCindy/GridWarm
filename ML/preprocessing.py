# preprocessing new
import numpy as np
import torch
import networkx as nx  # calculate distance between a node/edge and contingency locations

#colab
from os import getcwd
from sys import path
import os
import time
import multiprocessing as mp
print("Number of processors (CPUs): ", mp.cpu_count())
#
main_dir = getcwd() #os.path.dirname(getcwd()) #colab
path.append(main_dir + "/ML")
from GridSensitivities import LinearSystem #calculate Ybus

def makeYbus(contingency, bus_features, line_features, xfmr_features):
    lin_sys = LinearSystem(bus_features, line_features, xfmr_features)
    lin_sys.update_case(contingency)
    Ybus = lin_sys.Ybus.tocoo() #scipy sparse matrix: convert from csr matrix to coo matrix
    Ybus = torch.sparse_coo_tensor([Ybus.row.tolist(), Ybus.col.tolist()],
                                  torch.LongTensor(Ybus.data))#Ybus is a torch coo sparse matrix of size 2nbus by 2nbus
    return Ybus
def parse_sample(contingency, v_profile,
                 bus_features, line_features, xfmr_features, shunt_features=[]):
    """
    this func takes the raw sample data (attributes of ML_data class) as input
    and returns the parsed torch tensors as features for the ML model
    PS: shunt feature is actually not used in the current version of ML model
    INPUT:
     - contingency: a dict, e.g.  {'type': 'MadIoT','location':[1,2], 'parameter': [1.5, 1.3838348859970537]}
     - v_profile: a dict of dicts, e.g. v['pre'/'post'][bus num]= [vr, vi]
     - bus_features, line_features, xfmr_features, shunt_features: list of features, e.g. bus_features[ind] = [busnum, f1, f2, f3, f4...]
    OUTPUT: (dtype: double)
     - feat_i: torch tensor (n_activebus, n_raw_node_features), raw node features
     - feat_ni: torch tensor (n_activebus, n_raw_node_features), sum of neighboring node features
     - feat_ne: torch tensor (n_activebus, n_raw_edge_features), sum of features at neighbor edges of node i
     - feat_di: torch tensor (n_activebus,), distance(node,contingency)
     - feat_e: torch tensor (n_activeedge,n_raw_edge_features), raw edge features
     - feat_st: torch tensor (n_activeedge,2*n_raw_node_features), feature at end nodes of edge
     - feat_de: torch tensor (n_activeedge,), distance(edge, contingency)
     - bus_nums: (n_activebus) a torch vector of active bus numbers
     - if_ZIs: (n_activebus) a torch vector of "whether or not each active bus is ZI",
     - edge_st_nums: torch tensor (n_activebbus, 2), edge_st_nums[e] = [from_bus_num, to_bus_num]
     - Ybus: a torch sparse_coo_tensor matrix of Ybus, stamped by line and xfmr (no shunt)
    """
    n_nodes = len(v_profile['pre'].keys()) #n active nodes before contingency
    n_active_nodes = len(v_profile['post'].keys()) #n_active_nodes after contingency
    removed_buses = list(set(v_profile['pre'].keys()) - set(v_profile['post'].keys()))
    n_edges = len(line_features) + len(xfmr_features)
    if contingency['type']=='MadIoT':
        removed_edges = []
    else:
        raise NotImplementedError
    #
    feat_i = torch.tensor(bus_features)[:,1:]
    bus_nums = torch.tensor(bus_features)[:,0]
    if_ZIs = torch.tensor([torch.sum(torch.abs(feat[4:6]))<1e-6 for feat in feat_i])
    #
    feat_e = torch.cat((torch.tensor(line_features)[:,2:5],torch.tensor(xfmr_features)[:,2:5]),dim=0)
    edge_st_nums = torch.cat((torch.tensor(line_features)[:,0:2],torch.tensor(xfmr_features)[:,0:2]),dim=0)
    feat_st = torch.stack([torch.cat((feat_i[bus_nums==s],feat_i[bus_nums==t]),dim=1).view(-1) for s,t in edge_st_nums])
    #
    feat_ni = torch.zeros((n_nodes,10)).double()
    feat_ne = torch.zeros((n_nodes,3)).double()
    for e, (s,t) in enumerate(edge_st_nums):
        if not [int(s),int(t)] in removed_edges:
            feat_ni[bus_nums==s] = feat_ni[bus_nums==s] + feat_i[bus_nums==t]
            feat_ni[bus_nums==t] = feat_ni[bus_nums==t] + feat_i[bus_nums==s]
            feat_ne[bus_nums==s] = feat_ne[bus_nums==s] + feat_e[e]
            feat_ne[bus_nums==t] = feat_ne[bus_nums==t] + feat_e[e]
    #distance
    G = nx.Graph()
    G.add_nodes_from([int(b) for b in bus_nums])
    G.add_edges_from([(int(s),int(t)) for s,t in edge_st_nums])
    islanded_buses = list(nx.isolates(G))
    # if islanded_buses:
    #     print(G, islanded_buses)
    if contingency['type']=='MadIoT':
        contingency_buses = contingency['location']
    else:
        raise NotImplementedError
    shortest_path_dict = dict(nx.all_pairs_shortest_path_length(G))
    # distance(bus, contingencybuses)
    feat_di = torch.zeros(n_nodes).double()
    for ib,b in enumerate(bus_nums):
        b = int(b)
        if b in islanded_buses:
            feat_di[ib] = 1000
        else:
            if b not in contingency_buses:
                contingency_buses_trimmed = list(set(contingency_buses).intersection(shortest_path_dict[int(b)].keys())) #list(set(contingency_buses)-set(islanded_buses))
                if contingency_buses_trimmed:
                    feat_di[ib] = np.min([shortest_path_dict[int(b)][c] for c in contingency_buses_trimmed])
    # distance(edge, contingencybuses)
    feat_de = torch.zeros(n_edges).double()
    for ie, (s,t) in enumerate(edge_st_nums):
        feat_de[ie] = max(feat_di[bus_nums==s],feat_di[bus_nums==t])
    feat_di = feat_di*0.2
    feat_de = feat_de*0.2
    # domain knowledge:  Ybus for GaussianMRF_augY
    Ybus = torch.tensor([]) #Ybus = makeYbus(contingency, bus_features, line_features, xfmr_features)
    # truncate feat to only include active components
    active_bus_idx = [(int(bus) not in removed_buses) for bus in bus_nums]
    active_edge_idx = [([int(s),int(t)] not in removed_edges) for s,t in edge_st_nums]
    #
    feat_i, feat_ni, feat_ne = feat_i[active_bus_idx], feat_ni[active_bus_idx], feat_ne[active_bus_idx]
    bus_nums, if_ZIs = bus_nums[active_bus_idx], if_ZIs[active_bus_idx]
    feat_e, feat_st = feat_e[active_edge_idx], feat_st[active_edge_idx]
    edge_st_nums = edge_st_nums[active_edge_idx]
    feat_di, feat_de = feat_di[active_bus_idx], feat_de[active_edge_idx]
    return feat_i, feat_ni, feat_ne, feat_di, feat_e, feat_st, feat_de, bus_nums, if_ZIs, edge_st_nums, Ybus

class MyDataset(torch.utils.data.Dataset):
    #preprocessed features for use in ML model
    def __init__(self, Data, use_parallel = True): # test_order is the csv similar to what you used in hw1
        self.N = len(Data)
        # parallel computing to parse data? use_parallel=True?
        self.X, self.Y = self.parse_data(Data, use_parallel)
        assert(len(self.X) == len(self.Y))
        print("dataset initialized with %d samples"%self.N)
        # extra things needed for our ML model
        self.bus2name_book = Data[0].bus2name_book
        #self.name2bus_book = Data[0].name2bus_book
        self.BusNumList = list(self.bus2name_book.keys())
        #pointer, no need if using data loader
        self.pointer = 0
        self.if_end = False  # if the pointer is at the end of dataset
    def __len__(self):
        # need this if using dataloader
        return len(self.X)
    def __getitem__(self, ind):
        # need this if using data loader
        x = self.X[ind]
        y = self.Y[ind]
        return x, y

    def parse_data(self, Data, use_parallel = True):
        # parallel computing to parse data?
        if (mp.cpu_count() == 1) and use_parallel:
            #don't use parallel if there isn't enough cpus
            use_parallel = False
        X = []
        N = len(Data)
        start = time.time()
        if use_parallel:
            Ncpu = max(mp.cpu_count() - 2, 2)
            if N>500:
                print(Ncpu, " CPUs used for parallel preprocessing")
            pool = mp.Pool(Ncpu)
            X = pool.starmap(parse_sample, [(sample.contingency,
                                                  sample.v,
                                                  sample.x['bus feature'],
                                                  sample.x['line feature'],
                                                  sample.x['xfmr feature'],
                                                  sample.x['shunt feature']) for sample in Data])
            pool.close()
            pool.join()
            end = time.time()
            if N>500:
                print("%d samples parsed, %.2f seconds used"
                  % (len(Data), end - start),
                  end = "", flush=True)
        else:
            for sample in Data:
                X.append(parse_sample(sample.contingency,
                                           sample.v,
                                           sample.x['bus feature'],
                                           sample.x['line feature'],
                                           sample.x['xfmr feature'],
                                           sample.x['shunt feature']))
                if (N>500) and (len(X) % (max(round(0.01 * N), 10)) == 0):
                    end = time.time()
                    print('\r parsing input features %d/%d, %.2f spent' % (len(X), N, end - start),
                          end="", flush=True)
                    # print("parsing input features %d/%d"%(len(self.X),self.N))
        # Y: ground truth /label
        if N> 500:
            print("parsing labels          ")
        Y = [torch.tensor([sample.v['post'][b] for b in sample.v['post']]) for sample in Data]
        return X, Y

    def add_data(self, Data, use_parallel=True):
        #parse data and update self.X, self.Y, self.N
        X, Y = self.parse_data(Data, use_parallel)
        self.X.extend(X)
        self.Y.extend(Y)
        self.N = len(self.Y)
        assert (len(self.X) == len(self.Y))

    def reset(self):
        # no need this if using data loader
        self.pointer = 0  # idx of sample
        self.if_end = False  # if have already used all samples
        # print("reset dataset")
    def get_next_batch(self):
        # no need this if using dataloader
        # batchsize = 1
        x = []
        y = []
        if self.pointer < self.N:
            x = self.X[self.pointer]  # returns (my_nodes, my_edges, contingency, n_nodes, n_edges, v_pre)
            y = self.Y[self.pointer]  # returns a torch tensor of size Nbus by 2
        else:
            print('no more samples available, please reset dataset')
        self.pointer = self.pointer + 1
        self.if_end = (self.pointer >= self.N)
        return x, y