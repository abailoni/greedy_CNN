# --------------------------
# NETWORK 2:
# --------------------------
import time

import theano.tensor as T
from lasagne import layers


import greedyNET.greedy_utils as greedy_utils
import mod_nolearn.nets.segmNet as segmNet
from mod_nolearn.segmentFcts import pixel_accuracy_sigmoid


class Network2(object):
    '''
    This class contains the network that put together the LogRegr networks computed in the boosting procedure.

    Inputs:


    To solve:
     [x] I should pass at least one Regr, to get some informations automatically (num of filters of the first ConvLayer, etc...)
    '''
    def __init__(self, logRegr1, **kwargs):
        # -------------------------------------
        # Inherited parameters from logRegr1:
        # -------------------------------------
        self.num_classes = logRegr1.num_classes
        self.channels_image = logRegr1.channels_image
        self.xy_input = logRegr1.xy_input
        self.imgShape = logRegr1.imgShape
        self.eval_size = logRegr1.eval_size
        self.filter_size_convRegr = logRegr1.net.filter_size
        # Input processing:
        self.fixed_previous_layers = logRegr1.fixed_previous_layers
        self.DCT_size = logRegr1.DCT_size
        self.processInput = logRegr1.processInput

        # -------------------------------------
        # Specific parameters:
        # -------------------------------------
        self.batch_size = kwargs.pop('batch_size', 100)
        self.batchShuffle = kwargs.pop('batchShuffle', True)
        self.filter_size_conv2 = kwargs.pop('filter_size', 7)
        self.num_nodes = kwargs.pop('num_nodes', 5)

        customBatchIterator = greedy_utils.BatchIterator_Greedy(
            batch_size=self.batch_size,
            shuffle=self.batchShuffle,
            processInput=self.processInput
        )

        # -------------------------------------
        # CONSTRUCT NETWORK:
        # -------------------------------------
        netLayers = [
            # layer dealing with the input data
            (layers.InputLayer, {
                'shape': (None, self.processInput.output_channels, self.xy_input[0], self.xy_input[1])}),
            (layers.Conv2DLayer, {
                'name': 'conv_fixedRegr',
                'num_filters': self.num_classes*self.num_nodes,
                'filter_size': self.filter_size_convRegr,
                'pad':'same',
                'nonlinearity': segmNet.sigmoid_segm}),
            (maskLayer,{
                'name': 'mask',
                }),
            (layers.Conv2DLayer, {
                'name': 'conv2',
                'num_filters': self.num_classes,
                'filter_size': self.filter_size_conv2,
                'pad':'same',
                'nonlinearity': segmNet.sigmoid_segm}),
        ]

        # Set the first layer as not trainable:
        # ...


        # Insert the weights of the first network:
        # ....


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
        print "\n\n---------------------------\nCompiling Network 2...\n---------------------------"
        tick = time.time()
        self.net.initialize()
        tock = time.time()
        print "Done! (%f sec.)\n\n\n" %(tock-tick)


    def insert_weights(self, LogRegr, num_node):
        # Insert weights
        # ...
        # Change mask
        pass
