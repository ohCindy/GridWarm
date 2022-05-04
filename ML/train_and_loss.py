#train and loss
from os import getcwd
from sys import path
import time

import os

main_dir = getcwd() #os.path.dirname(getcwd()) #colab
path.append(main_dir + "/ML")

import ML_global_vars
USE_SPARSE_TOOLBOX = ML_global_vars.USE_SPARSE_TOOLBOX

LR = ML_global_vars.LR  # 0.001 very slow for case500
STEPLR_SS = ML_global_vars.STEPLR_SS
STEPLR_GAMMA = ML_global_vars.STEPLR_GAMMA
MOMENTUM = ML_global_vars.MOMENTUM

NUM_EPOCH = ML_global_vars.NUM_EPOCH
SAVE_FREQ = ML_global_vars.SAVE_FREQ  # save moel data every xx epochs
START_EPOCH = ML_global_vars.START_EPOCH  # file name starts from epoch ?

OUTPUT_MODE = ML_global_vars.OUTPUT_MODE  # model predicts 'V' or 'dV', Lambda*V = eta; or Lambda*dV = eta
# OUTPUT_MODE = 'dV' #todo, bug: didn't output Vpost in the final prediction

RESUME = ML_global_vars.RESUME

if USE_SPARSE_TOOLBOX:
    import torch_sparse_solve
    from torch_sparse_solve import solve

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # activation functions
import time

from ML_Utils import TrainingRecorder
from ML_Utils import plot_evaluation

class GaussianMRF_logLikelihood_Loss(nn.Module):
    # negative log likelihood loss
    # only support dense matrix
    def __init__(self):
        super(GaussianMRF_logLikelihood_Loss, self).__init__()

    def forward(self, Lambda, eta, y):
        # node and edge potent
        # input Lambda should be a dense matrix of size 2n by 2n
        # eta is a dense matrix of size 2n
        y = y.double()
        detLambda = torch.det(Lambda)
        if detLambda < 0:
            Lambda = -Lambda
            eta = -eta
        # v_pred, _ = torch.solve(eta, Lambda)
        v_pred = torch.linalg.solve(Lambda, eta)  # 2Nbus by 1 tensor
        logL = y.matmul(Lambda.matmul(y)) / 2 - eta.view(-1).matmul(y) + eta.view(-1).matmul(
            v_pred.view(-1)) / 2 - torch.logdet(Lambda) / 2
        # or we can sum up node and edge potentials one by one
        return logL


# main
class GaussianMRF_MSE_loss(nn.Module):
    # equivalent to -loglikelihood loss for gaussianmrf
    def __init__(self):
        super(GaussianMRF_MSE_loss, self).__init__()

    def forward(self, Lambda, eta, y):
        y_pred = MAP_prediction(Lambda, eta)
        loss = torch.mean((y_pred - y) ** 2)
        return loss


# main: get prediction by MAP estimate

def MAP_prediction(Lambda, eta):
    """
    this function solves solution z from Lambda*z = eta,
    N: number of nodes
    if use_sparse_toolbox=True, input Lambda is a sparse_coo_tensor, sized 2d: (2N by 2N),
    input eta is a dense vector, sized 2d: (2N by 1)
    output v_pred is a 1d tensor of size 2N"""
    if not USE_SPARSE_TOOLBOX:
        # input Lambda is dense matrix
        # v_pred, _ = torch.solve(eta, Lambda)
        v_pred = torch.linalg.solve(Lambda, eta)
        v_pred = v_pred.view(-1)
    else:
        v_pred = solve(torch.stack([Lambda]), torch.stack([eta])).view(-1)
    return v_pred

# main
def calc_err(y_pred, y):
    mse = torch.mean((y_pred - y) ** 2)
    vmag_err = torch.stack([torch.abs(val[0] + 1j * val[1]) for val in y_pred.view(-1, 2)]) - torch.stack(
        [torch.abs(val[0] + 1j * val[1]) for val in y.view(-1, 2)])
    me_vmag = torch.mean(torch.abs(vmag_err))  # mean error of vmag
    ang_err = torch.stack([torch.angle(val[0] + 1j * val[1]) for val in y.view(-1, 2)]) - torch.stack(
        [torch.angle(val[0] + 1j * val[1]) for val in y_pred.view(-1, 2)])
    me_ang = torch.mean(torch.abs(ang_err))  # mean error of angle
    return mse, me_vmag, me_ang


def evaluate(data_name, model, dataset, criterion, output_all=False):
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    mses = []
    mes_vmag = []
    mes_ang = []
    losses = []
    model.eval()
    dataset.reset()

    ys = []
    y_preds = []
    with torch.no_grad():
        while not dataset.if_end:
            x, y = dataset.get_next_batch()  # y: tensor of vpost sized [Nbus, 2]
            y = y.view(-1)  # convert to a 1d tensor
            # get feature from x
            feat_i, feat_ni, feat_ne, feat_di, feat_e, feat_st, feat_de, bus_nums, if_ZIs, edge_st_nums, Ybus = x
            #
            feat_i = feat_i.to(device)
            feat_ni = feat_ni.to(device)
            feat_ne = feat_ne.to(device)
            feat_di = feat_di.to(device)
            feat_e = feat_e.to(device)
            feat_st = feat_st.to(device)
            feat_de = feat_de.to(device)
            bus_nums = bus_nums.to(device)
            if_ZIs = if_ZIs.to(device)
            edge_st_nums = edge_st_nums.to(device)
            #
            y = y.to(device)
            #get output from model
            Lambda, eta = model(feat_i, feat_ni, feat_ne, feat_di,
                                feat_e, feat_st, feat_de,
                                bus_nums, if_ZIs, edge_st_nums)
            # get loss
            if not USE_SPARSE_TOOLBOX:
                Lambda = Lambda.to_dense()
            loss = criterion(Lambda, eta, y)
            # get accuracy:
            y_pred = MAP_prediction(Lambda.detach(), eta.detach()) # y_pred: a 1d tensor of sized 2Nbus
            mse, me_vmag, me_ang = calc_err(y_pred, y)
            losses.append(loss)
            mses.append(mse)
            mes_vmag.append(me_vmag)
            mes_ang.append(me_ang)
            #
            ys.append(y.data.tolist())
            y_preds.append(y_pred.data.tolist())
            if dataset.pointer % max(round(0.1*dataset.N),100)==0:
                print('\r evaluate (%d/%d), err %.4f, %.4f'
                      %(dataset.pointer, dataset.N, me_vmag, me_ang), end="")

    model.train()

    loss = sum(losses) / len(losses)
    out_mse = sum(mses) / len(mses)
    out_me_vmag = sum(mes_vmag) / len(mes_vmag)
    out_me_ang = sum(mes_ang) / len(mes_ang)

    if output_all:
        plot_evaluation(np.concatenate(ys), np.concatenate(y_preds), data_name+model.c_type, None)
        return loss, out_mse, out_me_vmag, out_me_ang, ys, y_preds
        # output ys and y_preds are list of lists, len is Nsample, ys[i_sample] = 1d list of y
    else:
        return loss, out_mse, out_me_vmag, out_me_ang


# main
def train(case_name, model, criterion,
          train_dataset, val_dataset, test_dataset,
          num_epoch=NUM_EPOCH, learning_rate=LR, save_freq=SAVE_FREQ):
    # send model to cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # define optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=MOMENTUM)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEPLR_SS, gamma=STEPLR_GAMMA)
    # record the loss/acc/err during the training, so that we can plot curve
    recorder = TrainingRecorder(case_name, model.c_type, model.method_name)
    #
    start_epoch = START_EPOCH
    #
    if RESUME:
        try:
            checkpoint_name = getcwd() + '/ML_Output/%s/%s.pth' % (case_name+model.c_type, case_name + model.method_name)
            checkpoint = torch.load(checkpoint_name) #map_location=device
            start_epoch = checkpoint['epoch'] + 1
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            scheduler = checkpoint['scheduler']
            recorder = checkpoint['recorder']
            print('resumed a checkpoint')
            print("checkpoint scheduler: stepLR, step %d, gamma %f"
                  % (scheduler.step_size, scheduler.gamma))
            recorder.print_history(min(15, recorder.count)) #print the last 10 records of losses and errors
            
        except FileNotFoundError:
            print("cannot find checkpoint file to resume, train a new model...")
    # training:
    for epoch in range(start_epoch, num_epoch):
        new_step = 5
        new_gamma = 0.5
        if (epoch > 74) and scheduler.gamma>new_gamma:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=new_step, gamma=new_gamma)
            print('resized stepLR scheduler gamma=%f'
                  %(scheduler.gamma))
        start = time.time()
        train_loss = 0.0  # total train loss
        train_me_vmag = 0.0  # total vmag error
        train_me_ang = 0.0  # total angle error in one epoch
        train_dataset.reset()
        while not train_dataset.if_end:
            optimizer.zero_grad()
            #
            x, y = train_dataset.get_next_batch()  # y: tensor of vpost sized [Nbus, 2]
            #
            y = y.view(-1)  # convert to a 1d tensor
            # get feature from x
            feat_i, feat_ni, feat_ne, feat_di, feat_e, feat_st, feat_de, bus_nums, if_ZIs, edge_st_nums, Ybus = x
            #
            feat_i = feat_i.to(device)
            feat_ni = feat_ni.to(device)
            feat_ne = feat_ne.to(device)
            feat_di = feat_di.to(device)
            feat_e = feat_e.to(device)
            feat_st = feat_st.to(device)
            feat_de = feat_de.to(device)
            bus_nums = bus_nums.to(device)
            if_ZIs = if_ZIs.to(device)
            edge_st_nums = edge_st_nums.to(device)
            #
            y = y.to(device)
            #get output from model
            Lambda, eta = model(feat_i, feat_ni, feat_ne, feat_di,
                                feat_e, feat_st, feat_de,
                                bus_nums, if_ZIs, edge_st_nums)
            # get loss
            if not USE_SPARSE_TOOLBOX:
                Lambda = Lambda.to_dense()
            loss = criterion(Lambda, eta, y)
            # optimize and update model 1)zero grad above, 2) backward 3) step
            loss.backward()
            optimizer.step()
            # accumulate train loss
            train_loss = train_loss + loss.item()
            # get train err/accuracy:
            y_pred = MAP_prediction(Lambda.detach(), eta.detach())  # y_pred: a 1d tensor of sized 2Nbus
            mse, me_vmag, me_ang = calc_err(y_pred.detach(), y.detach())
            train_me_vmag = train_me_vmag + me_vmag.item()
            train_me_ang = train_me_ang + me_ang.item()
            if train_dataset.pointer % max(round(0.1*train_dataset.N),100)==0:
                end = time.time()
                print('\r Epoch %d (%d/%d, %.2f sec), train loss %f err |V| %.4f, ang %.4f'
                      %(epoch,
                        train_dataset.pointer,train_dataset.N,
                        end - start,
                        train_loss/(train_dataset.pointer),
                        train_me_vmag/(train_dataset.pointer),
                        train_me_ang/(train_dataset.pointer)), end="")
            # report metrics
            if train_dataset.pointer % train_dataset.N == 0:  # % min(100, train_dataset.N) == 0:
                # get val err
                val_loss, val_mse, val_me_vmag, val_me_ang = evaluate(case_name + 'val', model, val_dataset, criterion,
                                                                       False)
                #
                # scheduler.step
                cur_lr = optimizer.param_groups[0]['lr']
                if cur_lr > 1e-5:
                    scheduler.step()

                #
                end = time.time()
                # print
                recorder.update(epoch,
                                train_loss / train_dataset.N,
                                train_loss / train_dataset.N,
                                train_me_vmag / train_dataset.N, train_me_ang / train_dataset.N,
                                val_loss.item(),
                                val_mse.item(), val_me_vmag.item(), val_me_ang.item())
                print("\r Epoch%d (%.2fs), train loss %f err |V| %.4f, ang %.4f; test loss %f err |V| %.4f, ang %.4f, lr=%.5f "
                      % (epoch, end - start,
                                train_loss / train_dataset.N,
                                train_me_vmag / train_dataset.N, train_me_ang / train_dataset.N,
                                val_loss.item(), val_me_vmag.item(), val_me_ang.item(), cur_lr))

        if (epoch + 1) % save_freq == 0:
            model.save(case_name, epoch)
            recorder.plot_losscurve()
            recorder.plot_errorcurve()
            # save checkpoint:
            checkpoint_name = getcwd() + '/ML_Output/%s/%s.pth' % (case_name+model.c_type, case_name + model.method_name)
            checkpoint = {
                'epoch': epoch,
                'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'recorder': recorder}
            torch.save(checkpoint, checkpoint_name)
            print('saved checkpoint')


