# ------------------------------
# For the moment just a LogRegr
# ------------------------------
import time
import numpy as np
from copy import deepcopy

import theano.tensor as T
from lasagne import layers
from lasagne.layers import set_all_param_values, get_all_param_values
from lasagne.nonlinearities import rectify, identity
import lasagne.init


import greedyNET.greedy_utils as greedy_utils
import mod_nolearn.nets.segmNet as segmNet
from mod_nolearn.segmentFcts import pixel_accuracy_sigmoid


class convSoftmax_routine(object):
    '''
    This class contains the network that put together the LogRegr networks computed in the boosting procedure.

    Inputs and options:
        - instance of class Boost_LogRegr (mandatory): the first trained node. Many parameters of Net2 are inherited by this network
        - batch_size (100)
        - batchShuffle (True)
        - filter_size (7): the dimension of the filter of the second convolutional level
        - num_nodes (5): number of nodes trained in a boosting matter
        - usual NeuralNet parameters (update, learning_rate...)

    Names of the layers:
        - conv1
        - mask
        - conv2
    '''
    def __init__(self, regrNode1, **kwargs):
        # -------------------------------------
        # Inherited parameters from regrNode1:
        # -------------------------------------
        self.num_classes = regrNode1.num_classes
        self.input_filters = regrNode1.input_filters
        self.xy_input = regrNode1.xy_input
        self.eval_size = regrNode1.eval_size
        self.filter_size1 = regrNode1.filter_size1
        self.filter_size2 = regrNode1.filter_size2
        self.num_filters_regr = regrNode1.num_filters1
        # Input processing:
        self.previous_layers = regrNode1.previous_layers
        # self.fixed_previous_layers = regrNode1.fixed_previous_layers
        # self.channels_image = regrNode1.channels_image
        # self.DCT_size = regrNode1.DCT_size
        # self.imgShape = regrNode1.imgShape

        # -------------------------------------
        # Specific parameters:
        # -------------------------------------
        self.batch_size = kwargs.pop('batch_size', 100)
        self.batchShuffle = kwargs.pop('batchShuffle', True)
        self.num_nodes = kwargs.pop('num_nodes', 5)

        customBatchIterator = greedy_utils.BatchIterator_Greedy(
            batch_size=self.batch_size,
            shuffle=self.batchShuffle,
            previous_layers=self.previous_layers
        )

        # -------------------------------------
        # CONSTRUCT NETWORK:
        # -------------------------------------
        netLayers = [
            # layer dealing with the input data
            (layers.InputLayer, {
                'name': 'inputLayer',
                'shape': (None, self.input_filters, self.xy_input[0], self.xy_input[1])}),
            (layers.Conv2DLayer, {
                'name': 'conv1',
                'num_filters': self.num_nodes*self.num_filters_regr*self.num_classes,
                'filter_size': self.filter_size1,
                'pad':'same',
                'nonlinearity': rectify}),
            (MaskLayer,{
                'name': 'mask',
                'num_filters_regr': self.num_filters_regr,
                'num_classes': self.num_classes
                }),
            (layers.Conv2DLayer, {
                'name': 'conv2',
                'num_filters': self.num_classes,
                'filter_size': self.filter_size2,
                'pad':'same',
                'nonlinearity': identity}),
            # New node:
            (layers.Conv2DLayer, {
                'incoming': 'inputLayer',
                'name': 'conv1_newNode',
                'num_filters': self.num_filters_regr*self.num_classes,
                'filter_size': self.filter_size1,
                'pad':'same',
                'nonlinearity': rectify}),
            (layers.Conv2DLayer, {
                'name': 'conv2_newNode',
                'num_filters': self.num_classes,
                'filter_size': self.filter_size2,
                'pad':'same',
                'nonlinearity': identity}),
            (layers.ElemwiseMergeLayer, {
                'incomings': ['conv2', 'conv2_newNode'],
                'merge_function': T.add}),
            (layers.NonlinearityLayer,{
                'nonlinearity': segmNet.sigmoid_segm}),
        ]

        self.net = segmNet.segmNeuralNet(
            layers=netLayers,
            batch_iterator_train = customBatchIterator,
            batch_iterator_test = customBatchIterator,
            objective_loss_function = segmNet.binary_crossentropy_segm,
            scores_train = [('trn pixelAcc', pixel_accuracy_sigmoid)],
            scores_valid = [('val pixelAcc', pixel_accuracy_sigmoid)],
            y_tensor_type = T.ftensor3,
            eval_size=self.eval_size,
            regression = True,
            **kwargs
        )

        # Set the first layer as not trainable:
        self.net._output_layer = self.net.initialize_layers()
        self.net.layers_['conv1'].params[self.net.layers_['conv1'].W].remove('trainable')
        self.net.layers_['conv1'].params[self.net.layers_['conv1'].b].remove('trainable')
        self.net.layers_['conv1_newNode'].params[self.net.layers_['conv1_newNode'].W].remove('trainable')
        self.net.layers_['conv1_newNode'].params[self.net.layers_['conv1_newNode'].b].remove('trainable')


        # print "\n\n---------------------------\nCompiling Network 2...\n---------------------------"
        # tick = time.time()
        # self.net.initialize()
        # tock = time.time()
        # print "Done! (%f sec.)\n\n\n" %(tock-tick)


        # Insert the weights of the first network:
        self.insert_weights(regrNode1)


    def insert_weights(self, regr):
        '''
        Structure of parameters:
         - W1: (num_classes*num_filters_regr*num_nodes, num_inputs, filter_length1)
         - b1: (num_classes*num_filters_regr*num_nodes, )
         - W2: (num_classes, num_classes*num_filters_regr*num_nodes, filter_length2)
         - b2: (num_classes,)
        '''
        # Update mask:
        self.net.layers_['mask'].add_node()
        actNode = self.net.layers_['mask'].active_nodes
        self.active_nodes = actNode

        # ------------------
        # Get weights:
        # ------------------
        W1, b1, maskParam, W2, b2 = layers.get_all_param_values(self.net.layers_['conv2'])
        oldNode_W1, oldNode_b1, oldNode_W2, oldNode_b2 = layers.get_all_param_values(self.net.layers_['conv2_newNode'])
        reg_W1, reg_b1, reg_W2, reg_b2 = layers.get_all_param_values(regr.net.layers_['conv2'])
        if actNode!=1:
            # --------------------
            # Update main part:
            # --------------------
            nNodes = self.num_classes*self.num_filters_regr
            start = nNodes*(actNode-2)
            stop = nNodes*(actNode-1)
            slice_weights = slice(start,stop)
            W1[slice_weights,:,:], b1[slice_weights] = oldNode_W1, oldNode_b1
            # For the moment I don't touch b2... Not sure about this...
            W2[:,slice_weights,:], b2 = oldNode_W2, b2+oldNode_b2
            layers.set_all_param_values(self.net.layers_['conv2'], [W1, b1, maskParam, W2, b2])
        # --------------------
        # Update new node:
        # --------------------
        oldNode_W1, oldNode_b1, oldNode_W2, oldNode_b2 = reg_W1, reg_b1, reg_W2, reg_b2
        layers.set_all_param_values(self.net.layers_['conv2_newNode'], [oldNode_W1, oldNode_b1, oldNode_W2, oldNode_b2])

        if actNode!=1:
            # Set layer conv2 not-trainable:
            self.net.layers_['conv2'].params[self.net.layers_['conv2'].W].remove('trainable')
            self.net.layers_['conv2'].params[self.net.layers_['conv2'].b].remove('trainable')
            self.net.initialize()

    def activate_nodes(self):
        '''
        Update the mask and make all nodes previous to the last one added trainable
        '''
        # Set layer conv2 trainable:
        self.net.layers_['conv2'].params[self.net.layers_['conv2'].W].add('trainable')
        self.net.layers_['conv2'].params[self.net.layers_['conv2'].b].add('trainable')
        self.net.initialize()


    # def clone(self,**kwargs):
    #     '''
    #     Options:
    #         - reset (True): for resetting the weights of the new Net

    #     Return the cloned object.
    #     '''
    #     if self.xy_input is not (None, None):
    #         raise Warning("Cloning with xy-image-inputs already set...")
    #     kwargs.setdefault('reset', True)
    #     kwargs.setdefault('setClassifier', False)
    #     newObj = deepcopy(self)
    #     if kwargs['reset']:
    #         # Reset some stuff:
    #         if newObj.net.verbose:
    #             # newObj.net.on_epoch_finished.append(PrintLog())
    #             pass
    #         newObj.net.train_history_[:] = []
    #         newObj._reset_weights()
    #     return newObj

    # def _reset_weights(self):
    #     W_regr, b_regr, mask, W_conv2, b_conv2 = get_all_param_values(self.net.layers_['conv2'])
    #     glorot, constant = lasagne.init.GlorotNormal(), lasagne.init.Constant()
    #     new_weights = [glorot.sample(W_regr.shape), constant.sample(b_regr.shape), mask, glorot.sample(W_conv2.shape), constant.sample(b_conv2.shape)]
    #     set_all_param_values(self.net.layers_['convLayer'], new_weights)
    #     self.net.layers_['mask'].active_nodes = 0
    #     self.active_nodes = 0

class MaskLayer(layers.Layer):
    '''
    --------------------------
    Subclass of lasagne.layers.Layer:
    --------------------------

    The received input should be in the form: (N, num_classes*num_nodes, dim_x, dim_y)

    Inputs:
     - num_filters_regr (5)
     - num_classes (1)

    '''
    # def __init__(self, incoming, num_units, W=lasagne.init.Normal(0.01), **kwargs):
    #     super(DotLayer, self).__init__(incoming, **kwargs)
    #     num_inputs = self.input_shape[1]
    #     self.num_units = num_units
    #     self.W = self.add_param(W, (num_inputs, num_units), name='W')

    def __init__(self, incoming, *args, **kwargs):
        # self.num_nodes = kwargs.pop('num_nodes', 5)
        self.num_filters_regr = kwargs.pop('num_filters_regr', 5)
        self.num_classes = kwargs.pop('num_classes', 1)
        super(MaskLayer, self).__init__(incoming, *args, **kwargs)
        self.active_nodes = 0
        # For the moment, an array such that: (deprecated)
        #   - the first element is actNods*nClas
        #   - the second is nClas*self.num_nodes
        self.active_nodes_slice = self.add_param(np.ones(4, dtype=np.int8), (4,), name='active_nodes_slice', trainable=False, regularizable=False)

    def add_node(self):
        '''
        Add one node and make all the others not trainable.
        '''
        self.active_nodes += 1
        actNods, nClas, nRegNodes = self.active_nodes, self.num_classes, self.num_filters_regr
        self.active_nodes_slice.set_value([0, 0,  (actNods-1)*nClas*nRegNodes, -1])

    # def activate_nodes(self):
    #     '''
    #     Make all nodes previous to the last one added trainable.
    #     '''
    #     actNods, nClas, nRegNodes = self.active_nodes, self.num_classes, self.num_filters_regr
    #     self.active_nodes_slice.set_value([0,0,actNods*nClas*nRegNodes, -1])


    def get_output_for(self, input, **kwargs):
        mod_input = T.set_subtensor(input[:,self.active_nodes_slice[0]:self.active_nodes_slice[1],:,:], 0.)
        return T.set_subtensor(mod_input[:,self.active_nodes_slice[2]:self.active_nodes_slice[3],:,:], 0.)




