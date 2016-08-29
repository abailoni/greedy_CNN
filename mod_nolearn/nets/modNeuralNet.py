
from time import time
import numpy as np
import matplotlib.pyplot as plt

from lasagne.layers import get_all_param_values

from nolearn.lasagne import NeuralNet
import mod_nolearn.utils as utils

from mod_nolearn.visualize import plot_fcts_show, plot_fcts

class AdjustVariable(object):
    def __init__(self, name, start=0.03, mode='linear', **kwargs):
        self.name = name
        self.start = start
        self.ls = None
        self.mode = mode
        self.decay_rate = kwargs.pop('decay_rate', None)
        self.stop = kwargs.setdefault('stop', None)

    def __call__(self, nn, train_history):
        epoch = train_history[-1]['epoch']
        if self.stop:
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

    def __call__(self, net, train_history, *args):
        self.iterations += 1
        mode = self.mode
        filename = self.filename
        # basename = filename[:-7] # remove .pickle

        if mode=="on_batch_finished":
            new_epoch = len(train_history)
            self.sub_iter = self.sub_iter+1 if self.epoch==new_epoch else 0
            self.epoch = new_epoch

        if mode=="on_training_finished":
            utils.pickle_model(net, filename)
        elif self.iterations%self.every==0:
            if self.only_best:
                this_loss = train_history[-1]['train_loss']
                best_loss = min([h['train_loss'] for h in train_history])
                if this_loss > best_loss:
                    return
            if mode=="on_epoch_finished":
                # utils.pickle_model(net, "%s_epoch_%d.pickle" %(basename, len(train_history)))
                utils.pickle_model(net, filename)
            # elif mode=="on_batch_finished":
            #     utils.pickle_model(net, "%s_epoch_%d-%d.pickle" %(basename, len(train_history), self.sub_iter))


from collections import OrderedDict
# from tabulate import tabulate

class save_train_history(object):
    def __init__(self, filename, every=1):
        self.filename = filename
        self.iterations = 0
        self.every = every
        self.out = []

    def __call__(self, net, train_history_):
        self.iterations += 1
        self.out.append(self.new_line(net, train_history_[-1]))
        if self.iterations%self.every==0:
            np.savetxt(self.filename, np.array(self.out), fmt='%.8f')


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

    def __call__(self, net, train_history, train_outputs):
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
        average = [np.average(a) for a in results]
        self.results.append([np.average(a) for a in average])

    def _write_file(self):
        results = np.array(self.results)
        np.savetxt(self.filename, results, fmt='%g')

    def _updatePlot(self):
        plt.clf()
        results = np.array(self.results)
        # train_scores = np.array(self.train_scores)
        plt.ion()
        plt.show()
        plot_fcts_show(self.iteration_history, [results[:,0]], labels=["Loss"], xyLabels=["Batch iterations", "Quantities"], log="y")
        plt.draw()
        plt.pause(0.001)

class track_weights_distrib(object):
    def __init__(self, **kwargs):
        self.every = kwargs.pop('every', None)
        self.layerName = kwargs.pop('layerName', None)
        self.pdfName = kwargs.pop('pdfName', None)

        self.iterations = 0
        self.results = []
        self.iteration_history = []

    def _write_file(self):
        results = np.array(self.results)
        np.savetxt(self.pdfName, results, fmt='%g')


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
        fig.savefig(self.pdfName)


def print_weight_distribution(net, layer_name=None):
    n_layers = len(net.layers)
    layers_names = [net.layers[i][1]['name'] for i in range(1,n_layers)]
    mean, std, weights = {}, {}, {}
    for name in layers_names:
        if "conv" in name:
            layer = net.layers_[name]
            W, _ = get_all_param_values(layer)
            mean[name], std[name], weights[name] = W.mean(), W.std(), W

    if layer_name:
        # print "Mean: %g; \tstd: %g" %(mean[layer_name], std[layer_name])
        return mean[layer_name],  std[layer_name]
    else:
        for name in mean:
            print "Layer %s: \tMean: %g; \tstd: %g" %(name, mean[name], std[name])








class modNeuralNet(NeuralNet):
    '''
    Modified version of NeuralNet (nolearn).

    ## FURTHER IDEAS ##

    - Accept separate training_data and test_data instead of dividing one array.

    -------------------------
    THINGS IMPLEMENTED
    -------------------------
    The function accepted as input on_batch_finished are of the kind:
        func(net, train_history_, train_outputs)
    and actually they are executed with a frequency 'numIter_subLog'.

    LOGS:
     - log_filename: the data in train_history are saved in a file (after each epoch)


    PICKLING THE MODEL:
     - pickle_filename (None): it will be used as baseline
     - pickleModel (None) accepts three options:
            - onTrainingFinished
            - atEpochFinished
            - atSubEpoch (the model is pickled with a frequency decided by 'numIter_subLog')


    SUB-EPOCHS LOGS:
    The saved info are loss and train_scores. A function for the weight ditribution is easily implemented (see nolearn_utils.py)
    For validation information use the pickled models.

     - numIter_subLog (None)
     - subLog_filename (None): if None the results are printed on screen. The file is updated only at the end of the epoch.
     - livePlot (False): live plot of the loss



    '''

    def __init__(self, *args, **kwargs):
        # Pickle:
        pickleModel_mode = kwargs.pop('pickleModel_mode', None)
        pickle_filename = kwargs.pop('pickle_filename', None)
        pickle_frequency = kwargs.pop('pickle_frequency', 1)
        # Logs:
        log_filename = kwargs.pop('log_filename', None)
        log_frequency = kwargs.pop('log_frequency', 1)
        # Sub-epoch logs:
        numIter_subLog = kwargs.pop('numIter_subLog', None)
        subLog_filename = kwargs.pop('subLog_filename', None)
        livePlot = kwargs.pop('livePlot', False)
        # Tracking weights:
        trackWeights_freq = kwargs.pop('trackWeights_freq', None)
        trackWeights_layerName = kwargs.pop('trackWeights_layerName', None)
        trackWeights_pdfName = kwargs.pop('trackWeights_pdfName', None)

        kwargs.setdefault('on_batch_finished', [])
        kwargs.setdefault('on_epoch_finished', [])
        kwargs.setdefault('on_training_finished', [])
        if pickleModel_mode:
            kwargs[pickleModel_mode] += [pickle_model(pickleModel_mode, pickle_filename, every=pickle_frequency)]
        if log_filename:
            kwargs['on_epoch_finished'] += [save_train_history(log_filename, every=log_frequency)]
        if numIter_subLog:
            kwargs['on_batch_finished'] += [save_subEpoch_history(numIter_subLog,filename=subLog_filename, livePlot=livePlot)]
        if trackWeights_freq:
            kwargs['on_batch_finished'] += [track_weights_distrib(layerName=trackWeights_layerName, pdfName=trackWeights_pdfName, every=trackWeights_freq)]


        super(modNeuralNet, self).__init__(*args, **kwargs)



    def train_loop(self, X, y, epochs=None):
        '''
        Modified version of the original one.
        '''
        epochs = epochs or self.max_epochs
        X_train, X_valid, y_train, y_valid = self.train_split(X, y, self)

        on_batch_finished = self.on_batch_finished
        if not isinstance(on_batch_finished, (list, tuple)):
            on_batch_finished = [on_batch_finished]

        on_epoch_finished = self.on_epoch_finished
        if not isinstance(on_epoch_finished, (list, tuple)):
            on_epoch_finished = [on_epoch_finished]

        on_training_started = self.on_training_started
        if not isinstance(on_training_started, (list, tuple)):
            on_training_started = [on_training_started]

        on_training_finished = self.on_training_finished
        if not isinstance(on_training_finished, (list, tuple)):
            on_training_finished = [on_training_finished]

        epoch = 0
        best_valid_loss = (
            min([row['valid_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        best_train_loss = (
            min([row['train_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        for func in on_training_started:
            func(self, self.train_history_)

        num_epochs_past = len(self.train_history_)

        while epoch < epochs:
            epoch += 1

            train_outputs = []
            valid_outputs = []

            if self.custom_scores:
                custom_scores = [[] for _ in self.custom_scores]
            else:
                custom_scores = []

            t0 = time()

            batch_train_sizes = []
            for Xb, yb in self.batch_iterator_train(X_train, y_train):
                train_outputs.append(
                    self.apply_batch_func(self.train_iter_, Xb, yb))
                batch_train_sizes.append(len(Xb))
                for func in on_batch_finished:
                    func(self, self.train_history_, train_outputs) # MOD LINE

            batch_valid_sizes = []
            for Xb, yb in self.batch_iterator_test(X_valid, y_valid):
                valid_outputs.append(
                    self.apply_batch_func(self.eval_iter_, Xb, yb))
                batch_valid_sizes.append(len(Xb))

                if self.custom_scores:
                    y_prob = self.apply_batch_func(self.predict_iter_, Xb)
                    for custom_scorer, custom_score in zip(
                            self.custom_scores, custom_scores):
                        custom_score.append(custom_scorer[1](yb, y_prob))

            train_outputs = np.array(train_outputs, dtype=object).T

            # Here each col contains one type of results (for training we have
            # [loss_train, scores_train]).
            # Each row is one number for each batch. We have np.mean() because one type of data are the scores (no idea why schould we track their avareage...)
            # In other words he's compressing all the informations found in an epoch. Why...?
            # CAREFUL!!!!!! BEFORE THERE IS A TRANSPOSITION, SO BEFORE IT'S THE OPPOSITE!!!!!!!!!!!!!!!!!!!!!!!!
            train_outputs = [
                np.average(
                    [np.mean(row) for row in col],
                    weights=batch_train_sizes,
                    )
                for col in train_outputs
                ]

            if valid_outputs:
                valid_outputs = np.array(valid_outputs, dtype=object).T
                valid_outputs = [
                    np.average(
                        [np.mean(row) for row in col],
                        weights=batch_valid_sizes,
                        )
                    for col in valid_outputs
                    ]

            if custom_scores:
                avg_custom_scores = np.average(
                    custom_scores, weights=batch_valid_sizes, axis=1)

            if train_outputs[0] < best_train_loss:
                best_train_loss = train_outputs[0]
            if valid_outputs and valid_outputs[0] < best_valid_loss:
                best_valid_loss = valid_outputs[0]

            info = {
                'epoch': num_epochs_past + epoch,
                'train_loss': train_outputs[0],
                'train_loss_best': best_train_loss == train_outputs[0],
                'valid_loss': valid_outputs[0]
                if valid_outputs else np.nan,
                'valid_loss_best': best_valid_loss == valid_outputs[0]
                if valid_outputs else np.nan,
                'valid_accuracy': valid_outputs[1]
                if valid_outputs else np.nan,
                'dur': time() - t0,
                }

            if self.custom_scores:
                for index, custom_score in enumerate(self.custom_scores):
                    info[custom_score[0]] = avg_custom_scores[index]

            if self.scores_train:
                for index, (name, func) in enumerate(self.scores_train):
                    info[name] = train_outputs[index + 1]

            if self.scores_valid:
                for index, (name, func) in enumerate(self.scores_valid):
                    info[name] = valid_outputs[index + 2]

            self.train_history_.append(info)

            try:
                for func in on_epoch_finished:
                    func(self, self.train_history_)
            except StopIteration:
                break

        for func in on_training_finished:
            func(self, self.train_history_)



