import mod_nolearn.tuneHyper as tuneHyper
import numpy as np

import os
os.environ["THEANO_FLAGS"] = "exception_verbosity=high,device=gpu0"


# path= "tuning/tune_first_regr"
path= "test/test_first_node"
path= "test/test_first_node_filter_size"
path= "tuning/tune_first_regr3"
path= "tuning/tune_first_node2"

tuneHyper.scatter_plot(path,exclude=40)


def print_sublog(model, quantity, path_out):
    path = path_out+'model_%d/cnv_L0_G0/' %model
    # path = path_out+'model_%d/regr_L0G0N1/' %model
    data = np.loadtxt(path+'sub_log.txt')
    data = data.reshape((data.shape[0],-1))
    if 'acc' in quantity.lower():
        return [np.arange(data.shape[0]), data[:,1]]
    elif 'loss' in quantity.lower():
        return [np.arange(data.shape[0]), data[:,0]]

def print_log(model, quantity, path_out):
    path = path_out+'model_%d/cnv_L0_G0/' %model
    # path = path_out+'model_%d/regr_L0G0N1/' %model
    data = np.loadtxt(path+'log.txt')
    x = data[:,0]
    if 'loss' in quantity.lower():
        if 'train' in quantity.lower():
            return x, data[:,1]
        if 'val' in quantity.lower():
            return x, data[:,2]
    if 'acc' in quantity.lower():
        if 'train' in quantity.lower():
            return x, data[:,4]
        if 'val' in quantity.lower():
            return x, data[:,5]



# def print_log_quantity_reg(model, quantity, path_out):
#     path = path_out+'model_%d/regr_L0G0N1/' %model
#     data = np.loadtxt(path+'log.txt')
#     if quantity=="train acc":
#         return [data[:,0], data[:]]

# def print_loss_sublog_reg(model, quantity, path_out):
#     path = path_out+'model_%d/regr_L0G0N1/' %model
#     data = np.loadtxt(path+'sub_log.txt')
#     return [np.arange(data.shape[0]), data[:]]



# ------------------------------
# COMPARE MODEL' FUNCTIONS:
# ------------------------------
quantity = "Training loss"
plot_kwargs = {'xyLabels': ['Iterations/epochs', 'Loss'],
    'log' : 'y',
    'label_size': 10,
    'ticks_size': 5 }

tuneHyper.compare_stuff(path, print_sublog, quantity, exclude=40, plot_kwargs=plot_kwargs)


# ------------------------------
# PLOT MODEL_RESULTS:
# ------------------------------
quantities = ["Train loss", "Validation loss"]
plot_kwargs = {'xyLabels': ['Epochs','Loss'],
    'log' : 'y',
    'label_size': 10,
    'ticks_size': 10}
tuneHyper.analyse_model(path, print_log, quantities, 195416, plot_kwargs=plot_kwargs)

quantities = ["Train accuracy", "Validation accuracy"]
plot_kwargs['log'] = ''
plot_kwargs['xyLabels'] = ['Epochs', 'Pixel accuracy']
tuneHyper.analyse_model(path, print_log, quantities, 195416, plot_kwargs=plot_kwargs)

# quantities = ["Train loss sublog"]
# plot_kwargs['log'] = ''
# plot_kwargs['xyLabels'] = ['Iterations', 'Loss']
# tuneHyper.analyse_model(path, print_sublog, quantities, 190721, plot_kwargs=plot_kwargs)

