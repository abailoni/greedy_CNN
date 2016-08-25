import time
from copy import deepcopy, copy

import matplotlib.pyplot as plt

import theano.tensor as T
import lasagne.layers as layers
from lasagne.layers import set_all_param_values, get_all_param_values


import pretr_nets.vgg16 as vgg16

import mod_nolearn.nets.segmNet as segmNet
import mod_nolearn.segmentFcts as segmentFcts
import greedyNET.nets.logRegres as logRegr_routine
import greedyNET.nets.net2 as net2_routine


class greedyRoutine(object):
    def __init__(self, num_VGG16_layers, **kwargs):
        '''
        Initialize a network that just uses the first layers of VGG16.

        Inputs:
         - How many layers to keep of VGG16 (int).
         - eval_size (0.1)

        Be careful with Pooling.

        All the arguments are the usual for nolearn NeuralNet (segmNet)
        '''
        # ------------------
        # Compile network:
        # ------------------
        self.num_layers = 0
        self.num_VGG16_layers = int(num_VGG16_layers)
        self.eval_size =  kwargs.pop('eval_size',0.)
        self.layers = vgg16.nolearn_vgg16_layers()[:self.num_VGG16_layers+1]
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

        print "\n\n---------------------------"
        print "Compiling inputProcess...\n---------------------------"
        tick = time.time()
        self.net.initialize()
        tock = time.time()
        print "Done! (%f sec.)\n\n\n" %(tock-tick)


        # --------------------
        # Copy vgg16 weights:
        # --------------------
        self.net = vgg16.nolearn_insert_weights_vgg16(self.net,self.num_VGG16_layers)

        self.output_channels = self.net.layers[-1][1]['num_filters']
        self.last_layer_name = self.net.layers[-1][1]['name']

    def train_new_layer(self, (fit_routine_LogRegr, num_LogRegr, kwargs_logRegr) , (fit_routine_net2, num_net2, kwargs_net2)):

        # ------------------------------------------------
        # Parallelized loop: (not working for the moment)
        # ------------------------------------------------
        Nets2 = []
        for idx_net2 in range(num_net2):
            # -----------------------------------------
            # Fit first LogRegr:
            # -----------------------------------------
            self.logRegr = logRegr_routine.Boost_LogRegr(
                self.net,
                self.output_channels,
                **kwargs_logRegr
            )
            self.logRegr.net = fit_routine_LogRegr(self.logRegr.net)
            self.best_classifier = copy(self.logRegr.net)


            # -----------------------------------------
            # Initialize Net2:
            # -----------------------------------------
            self.conv = net2_routine.Network2(
                self.logRegr,
                num_nodes=num_LogRegr,
                **kwargs_net2
            )

            # -----------------------------------------
            # Boosting loop:
            # -----------------------------------------
            for num_node in range(1,num_LogRegr):
                # Fit new logRegr to residuals:
                if num_node==1:
                    # Avoid deep_copy for best_classifier:
                    self.logRegr = logRegr_routine.Boost_LogRegr(
                        self.net,
                        self.output_channels,
                        **kwargs_logRegr
                    )
                else:
                    self.logRegr = self.logRegr.clone(reset=True)
                self.logRegr.set_bestClassifier(self.best_classifier)
                # Not sure about this step...
                self.logRegr.net = fit_routine_LogRegr(self.logRegr.net)

                # Fit Net2:
                self.conv.insert_weights(self.logRegr)
                self.conv.net = fit_routine_net2(self.conv.net)

                # Update the best classifier:
                self.best_classifier = self.conv.net

            Nets2.append(copy(self.conv)) ## This should be ok

        # Add new layer:
        self._insert_new_layer(Nets2[0])


    def _insert_new_layer(self, net2):
        '''
        Insert new layer, recompile and insert old weights.

        All parameters are trainable by default.

        IT SHOULD BE UPDATED TAKING A LIST OF NET2 COMPUTED IN PARALLEL...
        '''
        # -----------------
        # Collect weights:
        # -----------------

        prevLayers_weights = get_all_param_values(self.net.layers_[self.last_layer_name])
        net2_weights = get_all_param_values(net2.net.layers_['conv_fixedRegr'])

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
        self.last_layer_name = self.net.layers[-1][1]['name']
        set_all_param_values(self.net.layers_[self.last_layer_name], prevLayers_weights+net2_weights)

        self.output_channels = self.net.layers[-1][1]['num_filters']




