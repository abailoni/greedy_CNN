import time

import theano.tensor as T
import lasagne.layers as layers
from lasagne.layers import set_all_param_values, get_all_param_values


import pretr_nets.vgg16 as vgg16

import mod_nolearn.nets.segmNet as segmNet
import mod_nolearn.segmentFcts as segmentFcts

class greedyNet(object):
    def __init__(self, **kwargs):
        '''
        Initialize a network that just uses the first layers of VGG16.

        All the arguments are the usual for nolearn NeuralNet (segmNet)
        '''
        # ------------------
        # Compile network:
        # ------------------
        self.num_layers = 0
        self.layers = vgg16.nolearn_vgg16_layers()
        fixed_kwargs = {
            'objective_loss_function': segmNet.binary_crossentropy_segm,
            'scores_train': [('trn pixelAcc', segmentFcts.pixel_accuracy_sigmoid)],
            'scores_valid': [('val pixelAcc', segmentFcts.pixel_accuracy_sigmoid)],
            'y_tensor_type': T.ftensor3,
            'eval_size': self.eval_size,
            'regression': True
        }
        self.net_kwargs = kwargs.copy()
        self.net_kwargs.update(fixed_kwargs)

        self.net = segmNet.segmNeuralNet(
            layers=self.layers,
            **self.net_kwargs
        )

        print "\n\n---------------------------\n"
        print "Compiling inputProcess...\n---------------------------"
        tick = time.time()
        self.net.initialize()
        tock = time.time()
        print "Done! (%f sec.)\n\n\n" %(tock-tick)


        # --------------------
        # Copy vgg16 weights:
        # --------------------
        self.net = vgg16.nolearn_insert_weights_vgg16(self.net)

    def insert_new_layer(self, net2):
        '''
        Insert new layer, recompile and insert old weights.

        All parameters are trainable by default.
        '''
        # -----------------
        # Collect weights:
        # -----------------
        prevLayers_weights = get_all_param_values(self.net.layers_['conv%d' %(self.num_layers)])
        net2_weights = get_all_param_values(net2.layers_['conv_fixedRegr'])

        # -----------------
        # Add new layer:
        # -----------------
        self.num_layers += 1
        self.layers = self.layers + [
            (layers.Conv2DLayer, {
                'name': 'conv%d' %(self.num_layers),
                'num_filters': net2.num_classes*net2.num_nodes,
                'filter_size': net2.filter_size_convRegr,
                'pad':'same',
                'nonlinearity': segmNet.sigmoid_segm}),
        ]

        # ------------------
        # Recompile network:
        # ------------------
        self.net = segmNet.segmNeuralNet(
            layers=self.layers,
            **self.net_kwargs
        )
        print "\n\n---------------------------"
        print "Compiling and adding new layer...\n---------------------------"
        tick = time.time()
        self.net.initialize()
        tock = time.time()
        print "Done! (%f sec.)\n\n\n" %(tock-tick)

        # --------------------
        # Insert old weights:
        # --------------------
        set_all_param_values(self.net.layers_['conv%d' %(self.num_layers)], prevLayers_weights+net2_weights)



