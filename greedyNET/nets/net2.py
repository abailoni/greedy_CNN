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
            (MaskLayer,{
                'name': 'mask',
                'num_nodes': 5,
                'num_classes': self.num_classes
                }),
            (layers.Conv2DLayer, {
                'name': 'conv2',
                'num_filters': self.num_classes,
                'filter_size': self.filter_size_conv2,
                'pad':'same',
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
        self.net.layers_['conv_fixedRegr'].params[self.net.layers_['conv_fixedRegr'].W].remove('trainable')
        self.net.layers_['conv_fixedRegr'].params[self.net.layers_['conv_fixedRegr'].b].remove('trainable')

        # Insert the weights of the first network:
        self.insert_weights(logRegr1, 0)

        print "\n\n---------------------------\nCompiling Network 2...\n---------------------------"
        tick = time.time()
        self.net.initialize()
        tock = time.time()
        print "Done! (%f sec.)\n\n\n" %(tock-tick)


    def insert_weights(self, LogRegr, num_node):
        # Update mask:
        self.net.layers_['mask'].update_mask()

        # ------------------
        # Update weights:
        # ------------------
        Net2_W, Net2_b = layers.get_all_param_values(self.net.layers_['conv_fixedRegr'])
        W, b = layers.get_all_param_values(LogRegr.net.layers_['convLayer'])
        # Write down the shape of W and b and then it should be clear...
        # ....
        layers.set_all_param_values(self.net.layers_['conv_fixedRegr'], [Net2_W, Net2_b])


class MaskLayer(layers.Layer):
    '''
    --------------------------
    Subclass of lasagne.layers.Layer:
    --------------------------

    The received input should be in the form: (N, num_classes*num_nodes, dim_x, dim_y)

    Inputs:
     - num_nodes (5)
     - num_classes (1)

    '''
    def __init__(self, *args, **kwargs):
        self.num_nodes = kwargs.pop('num_nodes', 5)
        self.num_classes = kwargs.pop('num_classes', 1)
        self.active_nodes = 0
        super(MaskLayer, self).__init__(*args, **kwargs)

    def update_mask(self):
        self.active_nodes += 1

    def get_output_for(self, input, **kwargs):
        actNods, nClas = self.active_nodes, self.num_classes
        zero_indices = slice(actNods*nClas,nClas*self.num_nodes)
        return T.set_subtensor(input[:,zero_indices,:,:], 0.)




