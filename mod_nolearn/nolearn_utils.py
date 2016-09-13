# -----------------------------------------
# -----------------------------------------
#
# Collection of classes and functions used to perform useful actions
# at the end of batch iterations, epoch or training.
#
#
# In particular:
#
#  - AdjustVariable:
#       update some parameter at the end of an epoch
#       (e.g. the learning rate)
#
#  - pickle_model
#       accept frequency and only_if_the_best options
#
#  - save_train_history:
#       put train_history_ varaible in txt a file
#
#  - save_subEpoch_history:
#       collect useful logs after each batch iterations. Given a certain
#       frequency, it saves them in a file or updates a live-plot of the
#       loss.
#
#  - check_badLoss:
#       check possible infinite/NaN loss and send a StopIteration to the
#       training process
#
#  - track_weights_distrib:
#       needs some fixes...
#
#  - print_weight_distribution:
#       ...?
#
# -----------------------------------------
# -----------------------------------------


import numpy as np
from collections import OrderedDict
import warnings
import matplotlib.pyplot as plt


from lasagne.layers import get_all_param_values

from mod_nolearn.visualize import plot_fcts_show, plot_fcts
import various.utils as utils


class AdjustVariable(object):
    def __init__(self, name, start=0.03, mode='linear', **kwargs):
        self.name = name
        self.start = start
        self.ls = None
        self.mode = mode
        self.decay_rate = kwargs.pop('decay_rate', None)
        self.stop = kwargs.setdefault('stop', None)
        self.iteration = 0

    def __call__(self, nn, train_history):
        self.iteration += 1
        # epoch = train_history[-1]['epoch']
        # Consiously intented not to be the epoch... (if we reset it..)
        epoch = self.iteration
        if self.stop:
            # if train_history[-1]['epoch']>=nn.max_epochs:
            #     epoch = epoch%nn.max_epochs
            if self.iteration>=nn.max_epochs:
                new_value = utils.float32(self.ls[-1])
            else:
                if self.ls is None:
                    self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
                new_value = utils.float32(self.ls[epoch - 1])
            getattr(nn, self.name).set_value(new_value)
        elif self.decay_rate:
            old_value = getattr(nn, self.name).get_value()
            if self.mode=='linear':
                new_value = old_value/(1.+self.decay_rate*epoch)
            elif self.mode=='log':
                new_value = old_value*np.exp(-self.decay_rate*epoch)
            new_value = utils.float32(new_value)
            getattr(nn, self.name).set_value(new_value)


class pickle_model(object):
    def __init__(self, mode, filename, every=1, **kwargs):
        '''
        Modes available:
            - on_training_finished
            - on_epoch_finished
            - on_batch_finished (deprecated)

        Use the parameter 'every' to decide the frequency.
        '''
        self.mode = mode
        self.filename = filename
        self.every = every
        self.iterations = 0
        self.epoch = 0
        self.sub_iter = -1
        self.only_best = kwargs.pop('only_best', True)
        self.logs_path = kwargs.pop('logs_path', 'logs/')

    def __call__(self, net, train_history, *args):
        self.iterations += 1
        mode = self.mode
        filename = self.logs_path+self.filename
        # basename = filename[:-7] # remove .pickle

        if mode=="on_batch_finished":
            new_epoch = len(train_history)
            self.sub_iter = self.sub_iter+1 if self.epoch==new_epoch else 0
            self.epoch = new_epoch

        if mode=="on_training_finished":
            utils.pickle_model(net, filename)
        elif self.iterations%self.every==0:
            if self.only_best:
                this_loss = train_history[-1]['valid_loss']
                best_loss = min([h['valid_loss'] for h in train_history])
                if this_loss > best_loss:
                    return
            if mode=="on_epoch_finished":
                utils.pickle_model(net, filename)



class save_train_history(object):
    def __init__(self, filename, every=1, **kwargs):
        self.filename = filename
        self.iterations = 0
        self.every = every
        self.out = []
        self.logs_path = kwargs.pop('logs_path', 'logs/')

    def __call__(self, net, train_history_):
        self.iterations += 1
        if len(train_history_)!=0:
            self.out.append(self.new_line(net, train_history_[-1]))
            if self.iterations%self.every==0:
                np.savetxt(self.logs_path+self.filename, np.array(self.out), fmt='%.8f')

    def new_line(self, nn, info):
        info_tabulate = OrderedDict([
            ('epoch', info['epoch']),
            ('trn loss', info['train_loss']),
            ('val loss', info['valid_loss']),
            ('trn/val', info['train_loss'] / info['valid_loss']),
            ])

        if not nn.regression:
            info_tabulate['valid acc'] = info['valid_accuracy']

        for name, func in nn.scores_train:
            info_tabulate[name] = info[name]

        for name, func in nn.scores_valid:
            info_tabulate[name] = info[name]

        if nn.custom_scores:
            for custom_score in nn.custom_scores:
                info_tabulate[custom_score[0]] = info[custom_score[0]]

        if 'Train IoU' in info:
            info_tabulate['Train IoU'] = info['Train IoU']
        if 'Valid IoU' in info:
            info_tabulate['Valid IoU'] = info['Valid IoU']
        info_tabulate['lrn_rate'] = nn.update_learning_rate.get_value()
        # info_tabulate['update_beta1'] = nn.update_beta1.get_value()
        info_tabulate['dur'] = info['dur']

        return [info_tabulate[name] for name in info_tabulate]




class save_subEpoch_history(object):
    def __init__(self, every=10, **kwargs):
        self.every = every
        self.filename = kwargs.pop('filename', None)
        self.livePlot = kwargs.pop('livePlot', False)

        self.iterations = 0
        self.results = []
        self.iteration_history = []
        self.logs_path = kwargs.pop('logs_path', 'logs/')


    def __call__(self, net, train_history, train_outputs):
        '''
        train_outputs is a list with dimension:
            (N_iterations, N_quantities)
        and each element is an array (of one or more values).
        '''
        self.iterations += 1
        if self.iterations%self.every==0 and len(train_outputs)>1:
            self.iteration_history.append(self.iterations)
            self._append_results(train_outputs)
            if self.filename:
                self._write_file()
            if self.livePlot:
                self._updatePlot()
            # else:
            #     self._print_on_screen()

    def _append_results(self, train_outputs):
        if len(train_outputs)>self.every:
            results = np.array(train_outputs[-self.every:], dtype=object).T
        else:
            results = np.array(train_outputs, dtype=object).T
        # Averaging over iterations: (results.shape=(quantities,iterations))
        average = [np.average(a) for a in results]
        self.results.append(average)

    def _write_file(self):
        results = np.array(self.results)
        np.savetxt(self.logs_path+self.filename, results, fmt='%g')

    def _updatePlot(self):
        plt.clf()
        results = np.array(self.results)
        # train_scores = np.array(self.train_scores)
        plt.ion()
        plt.show()
        plot_fcts_show(self.iteration_history, [results[:,0]], labels=["Loss"], xyLabels=["Batch iterations", "Quantities"], log="")
        plt.draw()
        plt.pause(0.001)





class track_weights_distrib(object):
    '''
    Change with nice filename...
    '''

    def __init__(self, **kwargs):
        self.every = kwargs.pop('every', None)
        self.layerName = kwargs.pop('layerName', None)
        self.pdfName = kwargs.pop('pdfName', None)

        self.iterations = 0
        self.results = []
        self.iteration_history = []
        self.logs_path = kwargs.pop('logs_path', 'logs/')


    def _write_file(self):
        results = np.array(self.results)
        np.savetxt(self.logs_path+self.pdfName, results, fmt='%g')


    def __call__(self, net, *args):
        self.iterations += 1
        if self.iterations%self.every==0:
            self.results.append(print_weight_distribution(net, layer_name=self.layerName))
            self.iteration_history.append(self.iterations)
            self._write_file()

    def _savePlot(self):
        results = np.array(self.results)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_fcts(ax,self.iteration_history, [results[:,0], results[:,1]], labels=["Mean", "Std deviation"], xyLabels=["Batch iterations", ""])
        fig.set_tight_layout(True)
        fig.savefig(self.logs_path+self.pdfName)



def print_weight_distribution(net, layer_name=None):
    n_layers = len(net.layers)
    layers_names = [net.layers[i][1]['name'] for i in range(1,n_layers)]
    mean, std, weights = {}, {}, {}
    for name in layers_names:
        if "conv" in name:
            layer = net.layers_[name]
            W, _ = get_all_param_values(layer)[-2:]
            mean[name], std[name], weights[name] = W.mean(), W.std(), W

    if layer_name:
        # print "Mean: %g; \tstd: %g" %(mean[layer_name], std[layer_name])
        return mean[layer_name],  std[layer_name]
    else:
        for name in mean:
            print "Layer %s: \tMean: %g; \tstd: %g" %(name, mean[name], std[name])





def check_badLoss(net, train_history, train_outputs):
    if np.isnan(train_outputs[-1][0]) or np.isinf(train_outputs[-1][0]):
        warnings.warn("Training stopped, infinite loss")
        raise StopIteration


