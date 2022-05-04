from sys import path
from os import getcwd
import numpy as np

path.append(getcwd() + "/ML")

from ML_data import ML_data
from ML_data import DataReader

from preprocessing import MyDataset
from cGRF import Model_cGRF

from train_and_loss import GaussianMRF_logLikelihood_Loss, GaussianMRF_MSE_loss
from train_and_loss import train
from train_and_loss import evaluate

import multiprocessing as mp
import time

import torch

#loading data
case_name = 'ACTIVSg2000'#'ACTIVSg500'#   #'ACTIVSg2000' #
c_type = 'MadIoT'  #'gen outage' # MadIoT ; substation outage ; line outage; gen outage
"""
existing methods:
'cGRF', 'cGRF-ZI',
'cGRF-PS', 'cGRF-PS-ZI'
"""
method_names = ['cGRF-PS']
use_saved_model = True

print("===============Data loading==============")
"""read data"""
data_dir = getcwd()+"/Data/"
data_file_name = data_dir+(case_name+c_type)+"_data.npy"
data_reader = DataReader(data_file_name) #init up data/set file names for this case
#data_reader.append_data_file(data_file_name[0:-4]+'1.npy')
try:
    train_set = data_reader.load_data('train')
    val_set = data_reader.load_data('val')
    test_set = data_reader.load_data('test')
except:
    print('need to split data and save')
    train_set, val_set, test_set = data_reader.split()

print("%d training samples, %d val samples, %d test samples"%(len(train_set),len(val_set),len(test_set)))

"""parse data for ML models"""
#prepare data into proper format for ML input: parse contingency, split node/edge features, calc distance,
try:
    train_dataset, val_dataset, test_dataset = \
        np.load(data_file_name[0:-8]+"processed.npy", allow_pickle=True)
    print('loaded processed data')
except:
    use_parallel = True
    block = 100
    if mp.cpu_count() == 1:
        use_parallel = False
    print("parsing test data")
    start = time.time()
    if use_parallel:
        test_dataset = MyDataset(test_set[0:min(100,len(test_set))], use_parallel=True)
        if len(test_set)> 100:
            for ind in range(100, len(test_set), block):
                test_dataset.add_data(test_set[ind:min(ind + block, len(test_set))])
                end = time.time()
                print("\r %d/%d val samples parsed by parallel preprocessing, %s seconds used"
                      % (min(ind + block, len(test_set)), len(test_set), end - start),
                      end="", flush=True)
            print("\r")
    else:
        test_dataset = MyDataset(test_set)
    assert (test_dataset.N == len(test_set))
    print("parsing training data")
    start = time.time()
    if use_parallel:
        train_dataset = MyDataset(train_set[0:min(100,len(train_set))], use_parallel=True)
        if len(train_set)>100:
            for ind in range(100, len(train_set), block):
                train_dataset.add_data(train_set[ind:min(ind+block,len(train_set))])
                end = time.time()
                print("\r %d/%d train samples parsed by parallel preprocessing, %s seconds used"
                      %(min(ind+block,len(train_set)), len(train_set), end-start),
                      end="", flush=True)
            print("\r")
    else:
        train_dataset = MyDataset(train_set)
    assert (train_dataset.N == len(train_set))
    print("parsing validation data")
    start = time.time()
    if use_parallel:
        val_dataset = MyDataset(val_set[0:min(100,len(val_set))], use_parallel=True)
        if len(val_set)>100:
            for ind in range(100, len(val_set), block):
                val_dataset.add_data(val_set[ind:min(ind + block, len(val_set))])
                end = time.time()
                print("\r %d/%d val samples parsed by parallel preprocessing, %s seconds used"
                      % (min(ind + block, len(val_set)), len(val_set), end - start),
                      end="", flush=True)
            print("\r")
    else:
        val_dataset = MyDataset(val_set)
    assert (val_dataset.N == len(val_set))
    np.save(data_file_name[0:-8]+"processed.npy",[train_dataset,val_dataset,test_dataset])
    print("saved processed data")
print("comfirm processed data %d train samples, %d val samples, %d test samples"
      %(train_dataset.N, val_dataset.N, test_dataset.N))
#define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
for method_name in method_names:
    print("============%s, %s, %s=============="%(case_name, c_type, method_name))
    model = Model_cGRF(c_type, method_name,
                                    apply_parameter_sharing=('PS' in method_name),
                                       BusNumList=train_dataset.BusNumList,
                                       apply_ZI = ('ZI' in method_name)).double().to(device)
    epoch2load = 99
    if use_saved_model:
        print("trying to find saved model")
        try:
            model.load(case_name, epoch=epoch2load)
        except:
            print("no such file, train new model")
            use_saved_model = False

    # define loss
    criterion = GaussianMRF_MSE_loss()
    # train new model if needed
    if not use_saved_model:
        #
        try:
            if ('ZI' in method_name):
                # -ZI methods init with nonZI counterpart
                model.init_weights(case_name, epoch2load=4, method_name=method_name[0:-3])
                """
                model_ps = Model_cGRF(c_type, method_name,
                                    apply_parameter_sharing=True,
                                       BusNumList=train_dataset.BusNumList,
                                       apply_ZI = ('ZI' in method_name)).double().to(device)
                model_ps.init_weights(case_name, epoch2load=74, method_name='cGRF-PS-ZI')
                model.copy_PS_model(model_ps)
                """
            # else:
            #     model.init_weights(case_name, epoch2load=4, method_name=method_name)
        except:
            print('no warm start model found, retrain from default init')
        # train
        print("-----%s, %s, %s training starts-----"%(case_name, c_type, method_name))
        num_epoch = epoch2load + 1
        train(case_name, model, criterion,
                    train_dataset, val_dataset, test_dataset,
                    num_epoch)
        model.load(case_name, epoch=epoch2load)

    # get results on test data
    loss, mse, me_vmag, me_ang, ys, ypreds = evaluate(case_name + 'val', model, val_dataset,
                                                          criterion,True)
    print("--total val |V| and ang error %f %f" % (me_vmag, me_ang))
    # fill ypreds into val_set and store the updated data
    data_reader.update_and_save(val_set, ypreds, method_name, 'val')
    #get results on test data
    print("-----%s, %s, %s inference starts-----" % (case_name, c_type, method_name))
    output_all = True
    loss, mse, me_vmag, me_ang, ys, ypreds = evaluate(case_name+'test', model, test_dataset,
                                                          criterion, True)
    print("--total test |V| and ang error %f %f"%(me_vmag,me_ang))
    #fill ypreds into test_set and store the updated data
    data_reader.update_and_save(test_set, ypreds, method_name, 'test')
    """
    #get results on train data
    loss, mse, me_vmag, me_ang, ys, ypreds = evaluate(case_name+'train', model, train_dataset,
                                                          criterion, True)
    #fill ypreds into train_set and store the updated data
    data_reader.update_and_save(train_set, ypreds, method_name, 'train')
    print("--total train |V| and ang error %f %f" % (me_vmag, me_ang))  
    """


