# traing logger, CV logger
# ploting functions
import matplotlib.pyplot as plt
import numpy as np

import os

FONTSIZE = 16
NBINS = 30
MARKERS = ['.', '.', '.', '.']
COLORS = ['blue', 'green', 'orange', 'black']

figure_output_dir = os.getcwd() + '/Figure'
if not os.path.exists(figure_output_dir):
    os.makedirs(figure_output_dir)


class TrainingRecorder():
    # record the loss curve / err curve during training
    def __init__(self, case_name, c_type, method_name):
        self.count = 0
        self.epoch_rec = []
        self.val_mse_rec = []
        self.val_loss_rec = []
        self.train_mse_rec = []
        self.train_loss_rec = []
        self.val_me_vmag_rec = []
        self.train_me_vmag_rec = []
        self.val_me_ang_rec = []
        self.train_me_ang_rec = []
        self.name = case_name  # name of figures: casename+methodname
        self.method_name = method_name  # name of method
        self.c_type = c_type

    def update(self, epoch, loss, mse, me_vmag, me_ang, val_loss, val_mse, val_me_vmag, val_me_ang):
        self.count = self.count + 1
        self.epoch_rec.append(epoch)
        #
        self.train_mse_rec.append(mse)
        self.train_loss_rec.append(loss)
        self.train_me_vmag_rec.append(me_vmag)
        self.train_me_ang_rec.append(me_ang)
        #
        self.val_mse_rec.append(val_mse)
        self.val_loss_rec.append(val_loss)
        self.val_me_vmag_rec.append(val_me_vmag)
        self.val_me_ang_rec.append(val_me_ang)
    def print_history(self, K=10):
        #print the last K records
        print("training history: last %d records:"%K)
        for i in range(self.count-K, self.count):
            print("Rec %d, train loss %f, err |V| %.4f, ang %.4f; test loss %f, err |V| %.4f, ang %.4f"
                  %(i,
                    self.train_loss_rec[i], self.train_me_vmag_rec[i], self.train_me_ang_rec[i],
                    self.val_loss_rec[i], self.val_me_vmag_rec[i], self.val_me_ang_rec[i]))
    def plot_losscurve(self):
        # plot loss curve
        fig = plt.figure()
        plt.plot(self.train_loss_rec)
        plt.plot(self.val_loss_rec)
        plt.legend(['Train', 'Test'])
        plt.ylabel('Loss')
        plt.xlabel('#iterations')
        plt.title(self.name + self.c_type + self.method_name + 'Loss curve')
        plt.savefig(figure_output_dir + '/' + self.name + self.c_type + self.method_name + ' Loss.png')
        plt.close(fig)
        print('Saved figure of loss curve')

    def plot_errorcurve(self):
        fig = plt.figure(figsize=(15, 3))
        plt.subplot(131)
        plt.plot(self.train_mse_rec)
        plt.plot(self.val_mse_rec)
        plt.legend(['Train', 'Test'])
        plt.ylabel('MSE')
        plt.xlabel('#iterations')
        plt.title(self.name + self.c_type + self.method_name + ' MSE')
        plt.subplot(132)
        plt.plot(self.train_me_vmag_rec)
        plt.plot(self.val_me_vmag_rec)
        plt.legend(['Train', 'Test'])
        plt.ylabel('mean |V| error (p.u.)')
        plt.xlabel('#iterations')
        plt.title('mean |V| error')
        plt.subplot(133)
        plt.plot(self.train_me_ang_rec)
        plt.plot(self.val_me_ang_rec)
        plt.legend(['Train', 'Test'])
        plt.ylabel('mean angle error (rad)')
        plt.xlabel('#iterations')
        plt.title('mean angle error')
        plt.savefig(figure_output_dir + '/' + self.name + self.c_type + self.method_name + ' error curve.png')
        plt.close(fig)
        print('Saved figure of error curve')


class CVRecorder():
    def __init__(self):
        self.count = 0


def plot_a_calibration(y, y_pred, xlbl='', ylbl='', ttl='', topic='', legend_names=None):
    # input y and ypred are np array
    if y.ndim > 1:
        N_model = y.shape[0]
    else:
        N_model = 1
    y = y.reshape(N_model, -1)
    y_pred = y_pred.reshape(N_model, -1)
    print(N_model, 'models to be plotted')
    # plot
    fig = plt.figure(figsize=(5, 3))
    for m in range(N_model):
        plt.plot(y[m], y_pred[m], marker=MARKERS[m], linestyle='None', alpha=0.75)
    if legend_names:
        plt.legend(legend_names)
    plt.plot([np.min(y) - 0.1, np.max(y) + 0.1], [np.min(y) - 0.1, np.max(y) + 0.1], '--')
    plt.ylabel(ylbl, fontsize=FONTSIZE)
    plt.xlabel(xlbl, fontsize=FONTSIZE)
    plt.title(ttl, fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE - 4)
    plt.yticks(fontsize=FONTSIZE - 4)
    plt.savefig(figure_output_dir + '/' + topic + ttl + '.png')
    plt.close(fig)
    print(N_model, 'models to be plotted, figure saved at ', figure_output_dir + '/' + topic + ttl + '.png')
    # plt.show()


def plot_a_hist(y, xlbl='', ylbl='', ttl='', topic='', legend_names=None):
    # legend_names is for legends, topic is for figure file name, usually use case_name
    # input y is np array
    if y.ndim > 1:
        N_model = y.shape[0]
    else:
        N_model = 1
    y = y.reshape(N_model, -1)

    fig = plt.figure(figsize=(5, 3))
    for m in range(N_model):
        plt.hist(y[m], density=True, bins=NBINS, alpha=0.75)
    if legend_names:
        plt.legend(legend_names)
    plt.xticks(fontsize=FONTSIZE - 4)
    plt.yticks(fontsize=FONTSIZE - 4)
    plt.ylabel(ylbl, fontsize=FONTSIZE)
    plt.xlabel(xlbl, fontsize=FONTSIZE)
    plt.title(ttl, fontsize=FONTSIZE)
    # plt.show()
    plt.savefig(figure_output_dir + '/' + topic + ttl + '.png')
    plt.close(fig)
    print(N_model, 'models to be plotted, figure saved at ', figure_output_dir + '/' + topic + ttl + '.png')


def plot_evaluation(y, y_pred, topic='', legend_names=None):
    # for one sample, plot the v_i - vpred_i for all bus i
    """input y,ypred are 1d or 2d list of size 2Nbus*Ncase"""
    y = np.array(y)
    y_pred = np.array(y_pred)
    if y.ndim > 1:
        N_model = y.shape[0]
    else:
        N_model = 1
        # calibration of vr,vi
    plot_a_calibration(y, y_pred, 'Predicted bus voltage (Vr and Vi)', 'True bus volage', 'Phasor Calibration Plot',
                       topic, legend_names)
    # distribution Vr,Vi error
    v_err = y_pred - y
    plot_a_hist(v_err, 'Error of real and imag voltage', 'density', 'Vr&Vi Error Distribution', topic, legend_names)
    #
    y_pred = y_pred.reshape(N_model, -1, 2)
    y = y.reshape(N_model, -1, 2)
    vmag_pred = np.abs(y_pred[:, :, 0] + 1j * y_pred[:, :, 1])
    vmag = np.abs(y[:, :, 0] + 1j * y[:, :, 1])
    angle_pred = np.angle(y_pred[:, :, 0] + 1j * y_pred[:, :, 1])
    angle = np.angle(y[:, :, 0] + 1j * y[:, :, 1])
    vmag_err = vmag_pred - vmag
    ang_err = angle_pred - angle
    # |V| and angle calibration plot
    plot_a_calibration(vmag, vmag_pred, 'True bus |V|', 'Predicted bus |V|', '|V| Calibration Plot', topic,
                       legend_names)
    plot_a_calibration(angle, angle_pred, 'True angle', 'Predicted angle', 'Angle Calibration Plot', topic,
                       legend_names)
    # |V|, angle error distribution plot
    # plot_a_hist(vmag_err,'|V| error (pu)','density','|V| Error Distribution',topic,legend_names)
    # plot_a_hist(ang_err,'angle error (rad)','density','Angle Error Distribution',topic,legend_names)