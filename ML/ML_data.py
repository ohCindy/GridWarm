import numpy as np
from random import shuffle


class ML_data():
    def __init__(self, case_name, name2bus_book, bus2name_book, manipulate, contingency, x, v, Niter):
        self.case_name = case_name
        self.name2bus_book = name2bus_book.copy()  # bus: bus num; name: bus name; name2bus_book[name]=busnum
        self.bus2name_book = bus2name_book.copy()
        self.manipulate = manipulate.copy()  # dict,
        # print("Ml data saved manipulate",self.manipulate)
        self.contingency = contingency.copy()  # {'type':'MadIoT','location':[],'parameter':[cload,cdroop]}
        self.x = x.copy()  # dict, x['bus/line/xfmr/shunt feature']
        self.v = v.copy()  # dict v[bus num] = [vreal, vimag]
        self.vpred = dict()  # self.vpred[method name] is a dictionary of predicted solutions, key:busnun, value[vr,vi], like v['post']
        self.Niter = Niter.copy()  # dict()
        self.Niter_pre = Niter['pre simu']  # save of copy of pre simulation
        # self.Niter['file'] = Niter
        # print("#of iterations with file data init:",Niter)


class DataReader():
    def __init__(self, data_file_name):
        # read data
        #data_file_name: ACTIVSg2000MadIoT_data.npy
        #data_file_names: a list of data file names
        self.data_file_names = [data_file_name]
        print(self.data_file_names)
        self.set_file_name = dict()
        self.set_file_name['train'] = data_file_name[0:-8] + 'train.npy'
        self.set_file_name['val'] = data_file_name[0:-8] + 'val.npy'
        self.set_file_name['test'] = data_file_name[0:-8] + 'test.npy'
    def append_data_file(self, data_file_name):
        self.data_file_names.append(data_file_name)
        print("added %s"%data_file_name)
    def load_data(self, data_name=None):
        if not data_name:
            Data = []
            for data_file_name in self.data_file_names:
                Data_part = np.load(data_file_name, allow_pickle=True).tolist()
                Data.extend(Data_part)
            print("data loaded")
        else:
            Data = np.load(self.set_file_name[data_name], allow_pickle=True).tolist()
            print('%s data loaded'%data_name)
        return Data

    def split(self, Nlimit=5000, r_train=0.8, r_val=0.1, r_test=None, save_data=True):
        # split and save the entire dataset
        Data = self.load_data()
        if Nlimit<len(Data):
            Data = Data[0:Nlimit]
        self.N = len(Data)  # length of total data
        ## split data into train validation test
        train_set, val_set, test_set = split_data(Data, r_train, r_val, r_test)
        if save_data:
            np.save(self.set_file_name['train'], train_set)
            np.save(self.set_file_name['val'], val_set)
            np.save(self.set_file_name['test'], test_set)
        print('data split and saved!')
        return train_set, val_set, test_set

    def update_and_save(self, Data, ypreds, method_name, data_name=None):
        # fill and save
        # ypreds is a list of lists, len=Nsample, each row is a 1d array of ypred
        # this function fills ypreds into sample.vpred[methodname] as a dictionary, just like sample.v['post']
        for i, ypred in enumerate(ypreds):
            # fill a 1d list ypred into dictionary key=busnum, value =[vr,vi]
            if not hasattr(Data[i], 'vpred'):
                Data[i].vpred = dict()
            Data[i].vpred[method_name] = dict()
            for ib, busnum in enumerate(Data[i].v['post'].keys()):
                Data[i].vpred[method_name][busnum] = ypred[2 * ib:(2 * ib + 2)]
        if not data_name:
            save_file_name = self.data_file_name
        else:
            save_file_name = self.set_file_name[data_name]
        np.save(save_file_name, Data)
        print('data filled with prediction and saved at', self.set_file_name[data_name])


def split_data(Data, r_train, r_val, r_test):
    N_train = round(r_train * len(Data))
    N_val = round(r_val * len(Data))
    N_test = len(Data) - N_train - N_val
    if r_test:
        N_test = min(N_test, round(r_test * len(Data)) - 1)
    # shuffle
    # shuffle(Data)
    # split
    train_set = Data[0:N_train]
    val_set = Data[N_train:N_train + N_val]
    test_set = Data[N_train + N_val:]
    return train_set, val_set, test_set
