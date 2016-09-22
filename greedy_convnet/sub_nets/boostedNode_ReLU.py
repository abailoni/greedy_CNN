'''
Modified version of boostedNode.py using directly a ReLU non-linearity instead
of going from sigmoid to ReLU
'''




from copy import deepcopy
import json

import theano.tensor as T
from lasagne import layers
from lasagne.nonlinearities import rectify
from lasagne.init import Normal


from mod_nolearn.segm.segm_utils import pixel_accuracy, softmax_segm
from mod_nolearn.segm import segmNeuralNet


from greedy_convnet.sub_nets.boostedNode import BatchIterator_boostRegr, categorical_crossentropy_segm_boost





class boostedNode_ReLU(object):
    def __init__(self,greedyLayer,**kwargs):
        info = deepcopy(kwargs)
        # --------------------------
        # Inherited by convSoftmax:
        # --------------------------
        self.xy_input = greedyLayer.xy_input
        self.filter_size1 = greedyLayer.filter_size1
        self.filter_size2 = greedyLayer.filter_size2
        self.input_filters = greedyLayer.input_filters
        self.num_filters1 = greedyLayer.num_filters1
        self.num_classes = greedyLayer.num_classes
        self.previous_layers = greedyLayer.previous_layers
        self.eval_size = greedyLayer.eval_size
        self.best_classifier = greedyLayer.net if not greedyLayer.net.layers_['mask'].first_iteration else None

        # -----------------
        # Own attributes:
        # -----------------
        kwargs.setdefault('name', 'boostedNode')
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
        info.pop('noReg_loss', None)
        for key in [key for key in info if 'update_' in key]:
            info[key] = info[key].get_value().item()
        json.dump(info, file(info['logs_path']+'/info-net.txt', 'w'))

