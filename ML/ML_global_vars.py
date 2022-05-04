# global vars
# BATCHSIZE = 1

START_EPOCH = 0 #training index start from epoch x?
NUM_EPOCH = 100 #train until epoch#x
SAVE_FREQ = 5  # save model data every xx epochs

LR = 0.001 #0.001 very slow for case500
STEPLR_SS = 5 #stepLR scheduler, step size
STEPLR_GAMMA = 0.99 #stepLR scheduler, gamma
#
MOMENTUM = 0.9 #if using SGD
#
OUTPUT_MODE = 'V' # model predicts 'V' or 'dV', Lambda*V = eta; or Lambda*dV = eta
# OUTPUT_MODE = 'dV' #todo, bug: didn't output Vpost in the final prediction
#
HIDDEN_DIM = 64 #width of NN 32 doesn't work well for 'dV' mode
N_HIDDEN= 1 #num of hidden layers
#64,1 works well for case30, case118, madiot,gaussianMRF, both with/wo PS

USE_SPARSE_TOOLBOX = True #uses torch_sparse_solve package, set to False if not having the package
RESUME = True #resume training

print("=======================Configurations===================")
print("- Save model every %d epochs"
      %(SAVE_FREQ))
print("- LR starts %f, stepLR scheduler step %d, gamma %f"
      %(LR, STEPLR_SS, STEPLR_GAMMA))
print("- NN architecture %d hidden layers, hidden dim = %d" %(N_HIDDEN, HIDDEN_DIM))
if USE_SPARSE_TOOLBOX:
    print("- Training uses sparse toolbox torch_sparse_solve")
else:
    print("- Training uses dense matrix format")

