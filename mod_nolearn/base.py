
from time import time
import numpy as np


from lasagne.objectives import aggregate
from lasagne.layers import get_output
from lasagne import regularization

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator

from mod_nolearn.nolearn_utils import *



class modBatchIterator(BatchIterator):
    '''
    It fixes a bug related to randomization
    '''
    def __call__(self, X, y=None):
        if self.shuffle and y is not None:
            self._shuffle_arrays([X, y] if y is not None else [X], self.random)
        self.X, self.y = X, y
        return self



class modObjective(object):
    def __init__(self,l2=0):
        self.l2 = l2

    def __call__(self, layers,
                  loss_function,
                  target,
                  aggregate=aggregate,
                  deterministic=False,
                  l1=0,
                  l2=0,
                  get_output_kw=None):
        # Mode:
        l2 = self.l2

        if get_output_kw is None:
            get_output_kw = {}
        output_layer = layers[-1]
        network_output = get_output(
            output_layer, deterministic=deterministic, **get_output_kw)
        loss = aggregate(loss_function(network_output, target))

        if l1:
            loss += regularization.regularize_layer_params(
                layers.values(), regularization.l1) * l1
        if l2:
            loss += regularization.regularize_layer_params(
                layers.values(), regularization.l2) * l2
        return loss




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





        # THIS SHOULD BE ALL UPDATED WITH NICE UPDATE METHODS...
        # E.g.: update pickleModel_mode, then set again content of 'on_something_end'

        Put attributes as proprieties...

    '''

    def __init__(self, *args, **kwargs):
        self.logs_path = kwargs.pop('logs_path', './logs/')
        self.name = kwargs.pop('name', 'segmNet')

        # Regularization:
        if not hasattr(self, 'L2'):
            self.L2 = kwargs.pop('L2', 0)
            if self.L2:
                kwargs['objective'] = modObjective(self.L2)


        # Pickle:
        pickleModel_mode = kwargs.pop('pickleModel_mode', None)
        pickle_filename = kwargs.pop('pickle_filename', self.name+'.pickle')
        pickle_frequency = kwargs.pop('pickle_frequency', 1)

        # Logs:
        log_frequency = kwargs.pop('log_frequency', None)
        log_filename = kwargs.pop('log_filename', 'log.txt')

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
            kwargs[pickleModel_mode] += [pickle_model(pickleModel_mode, pickle_filename, every=pickle_frequency, logs_path=self.logs_path)]
        if log_frequency:
            kwargs['on_epoch_finished'] += [save_train_history(log_filename, every=log_frequency, logs_path=self.logs_path)]
        if numIter_subLog:
            kwargs['on_batch_finished'] += [save_subEpoch_history(numIter_subLog,filename=subLog_filename, livePlot=livePlot, logs_path=self.logs_path)]
        if trackWeights_freq:
            kwargs['on_batch_finished'] += [track_weights_distrib(layerName=trackWeights_layerName, pdfName=trackWeights_pdfName, every=trackWeights_freq, logs_path=self.logs_path)]

        # Check when to exit the training loop:
        kwargs['on_batch_finished'] += [check_badLoss]


        super(modNeuralNet, self).__init__(*args, **kwargs)


    def update_logs_path(self, new_path):
        self.logs_path = new_path
        # List of classes with an attribute logs_path:
        Classes = (pickle_model, save_train_history, save_subEpoch_history, track_weights_distrib)
        for fun in self.on_batch_finished+self.on_epoch_finished+self.on_training_finished:
            if isinstance(fun, Classes):
                fun.logs_path = new_path
                # Temporary check for correcting previous shit...
                if hasattr(fun, 'filename'):
                    fun.filename = fun.filename.split('/')[-1]
                if hasattr(fun, 'pdfName'):
                    fun.pdfName = fun.pdfName.split('/')[-1]

    def update_AdjustObjects(self, new_objects):
        '''
        'new_objects' is a list of AdjustVariable instances to insert.
        '''
        delete = []
        for i, fun in enumerate(self.on_epoch_finished):
            if isinstance(fun, AdjustVariable):
                delete.append(fun)
        for fun in delete:
            self.on_epoch_finished.remove(fun)
        self.on_epoch_finished += new_objects

    def loss(self, X, y):
        '''
        Takes some data and return the averaged loss.
        '''
        valid_outputs, batch_valid_sizes = [], []
        for Xb, yb in self.batch_iterator_test(X,y):
                valid_outputs.append(
                    self.apply_batch_func(self.eval_iter_, Xb, yb))
                batch_valid_sizes.append(len(Xb))
        valid_outputs = np.array(valid_outputs, dtype=object).T
        return np.average( [np.mean(row) for row in valid_outputs[0]]  , weights=batch_valid_sizes)

    def train_loop(self, X, y, epochs=None):
        '''
        Modified version of the original one.

        The changes are highlighted in the comments.
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
            try: ### MOD
                for Xb, yb in self.batch_iterator_train(X_train, y_train):
                    train_outputs.append(
                        self.apply_batch_func(self.train_iter_, Xb, yb))
                    batch_train_sizes.append(len(Xb))
                    for func in on_batch_finished:
                        func(self, self.train_history_, train_outputs) # MOD LINE
            except StopIteration:
                break ### MOD

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

            # Check sublogs for details about train_outputs
            train_outputs = np.array(train_outputs, dtype=object).T

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


