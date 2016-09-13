from copy import deepcopy
import json

import theano.tensor as T
from lasagne import layers
from lasagne.nonlinearities import rectify
from lasagne.init import Normal


from mod_nolearn.segm.segm_utils import pixel_accuracy, softmax_segm
from mod_nolearn.segm import segmNeuralNet


from greedyConvNET.subNets.boostedNode import BatchIterator_boostRegr, categorical_crossentropy_segm_boost


class boostedNode_ReLU(object):
    '''
    TO BE COMPLETELY UPDATED
        - logs_path


    Options and inputs:
        - processInput (Required): istance of the class greedy_utils.processInput.
        - filter_size (7): requires an odd size to keep the same output dimension
        - best_classifier (None): best previous classifier
        - batch_size (100)
        - batchShuffle (True)
        - eval_size (0.1): decide the cross-validation proportion. Do not use the option "train_split" of NeuralNet!
        - xy_input (None, None): use only for better optimization. But then use the cloning process at your own risk.
        - all additional parameters of NeuralNet.
          (e.g. update=adam, max_epochs, update_learning_rate, etc..)

    Deprecated:
        - [num_classes now fixed to 2, that means one filter...]
        - imgShape (1024,768): used for the final residuals
        - channels_image (3): channels of the original image

    IMPORTANT REMARK: (valid only for Classification, not LogRegres...)
    in order to implement a boosting classification, the fit function as targets y should get a matrix with floats [0, 0, ..., 1, ..., 0, 0] instead of just an array of integers with the class numbers.

    To be fixed:
        - a change in the batch_size should update the batchIterator
        - add a method to update the input-processor
        - channels_input with previous fixed layers
        - check if shuffling in the batch is done in a decent way
    '''
    def __init__(self,convSoftmaxNet,**kwargs):
        info = deepcopy(kwargs)
        # --------------------------
        # Inherited by convSoftmax:
        # --------------------------
        self.xy_input = convSoftmaxNet.xy_input
        self.filter_size1 = convSoftmaxNet.filter_size1
        self.filter_size2 = convSoftmaxNet.filter_size2
        self.input_filters = convSoftmaxNet.input_filters
        self.num_filters1 = convSoftmaxNet.num_filters1
        self.num_classes = convSoftmaxNet.num_classes
        self.previous_layers = convSoftmaxNet.previous_layers
        self.eval_size = convSoftmaxNet.eval_size
        self.best_classifier = convSoftmaxNet.net if not convSoftmaxNet.net.layers_['mask'].first_iteration else None

        # -----------------
        # Own attributes:
        # -----------------
        kwargs.setdefault('name', 'boostRegr')
        self.init_weight = kwargs.pop('init_weight', 1e-3)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.batchShuffle = kwargs.pop('batchShuffle', True)

        # -----------------
        # Input processing:
        # -----------------
        if not self.xy_input[0] or not self.xy_input[1]:
            raise ValueError("Xy dimensions required for boosting weights")
        out_shape = (self.batch_size, self.xy_input[0], self.xy_input[1])
        objective_loss_function = categorical_crossentropy_segm_boost(out_shape)
        customBatchIterator = BatchIterator_boostRegr(
            objective_boost_loss=objective_loss_function,
            batch_size=self.batch_size,
            shuffle=self.batchShuffle,
            best_classifier=self.best_classifier,
            previous_layers=self.previous_layers
        )

        # -----------------
        # Building NET:
        # -----------------
        netLayers = [
            # layer dealing with the input data
            (layers.InputLayer, {'shape': (None, self.input_filters, self.xy_input[0], self.xy_input[1])}),
            (layers.Conv2DLayer, {
                'name': 'conv1',
                'num_filters': self.num_filters1,
                'filter_size': self.filter_size1,
                'pad':'same',
                'W': Normal(std=self.init_weight),
                'nonlinearity': rectify }),
            (layers.Conv2DLayer, {
                'name': 'conv2',
                'num_filters': self.num_classes,
                'filter_size': self.filter_size2,
                'pad':'same',
                'W': Normal(std=self.init_weight),
                'nonlinearity': softmax_segm}),
            # (UpScaleLayer, {'imgShape':self.imgShape}),
        ]

        self.net = segmNeuralNet(
            layers=netLayers,
            batch_iterator_train = customBatchIterator,
            batch_iterator_test = customBatchIterator,
            objective_loss_function = objective_loss_function,
            scores_train = [('trn pixelAcc', pixel_accuracy)],
            y_tensor_type = T.ltensor3,
            eval_size=self.eval_size,
            **kwargs
        )
        # print "\n\n---------------------------\nCompiling regr network...\n---------------------------"
        # tick = time.time()
        self.net.initialize()
        # tock = time.time()
        # print "Done! (%f sec.)\n\n\n" %(tock-tick)

        # -------------------------------------
        # SAVE INFO NET:
        # -------------------------------------
        info['filter_size1'] = self.filter_size1
        info['filter_size2'] = self.filter_size2
        info['input_filters'] = self.input_filters
        info['num_filters1'] = self.num_filters1
        info['num_classes'] = self.num_classes
        info['xy_input'] = self.xy_input
        info['eval_size'] = self.eval_size

        info.pop('update', None)
        info.pop('on_epoch_finished', None)
        info.pop('on_batch_finished', None)
        info.pop('on_training_finished', None)
        for key in [key for key in info if 'update_' in key]:
            info[key] = info[key].get_value().item()
        json.dump(info, file(info['logs_path']+'/info-net.txt', 'w'))

