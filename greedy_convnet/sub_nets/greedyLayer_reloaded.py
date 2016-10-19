import numpy as np
from copy import deepcopy
import json

import theano.tensor as T
from lasagne import layers
from lasagne.nonlinearities import rectify, identity

from lasagne.init import Normal

import mod_nolearn.segm.segm_utils as segm_utils
from mod_nolearn.segm import segmNeuralNet
from greedy_convnet import BatchIterator_Greedy


class greedyLayer_reload(object):
    def __init__(
            self,
            fixed_input_layers,
            layers_info,
            name_trained_layer,
            trained_layer_args,
            list_boost_filters,
            **kwargs):
        info = deepcopy(kwargs)
        # -----------------
        # General attributes:
        # -----------------
        self.init_weight = trained_layer_args.pop('init_weight', 1e-3)
        self.filter_size1 = trained_layer_args.pop('filter_size', 7)
        self.filter_size2 = self.filter_size1
        self.num_filters1 = trained_layer_args.pop('num_filters', 5)

        self.fixed_input_layers = fixed_input_layers

        kwargs['name'] = "greedy_"+name_trained_layer
        self.num_classes = kwargs.pop('num_classes', 2)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.batchShuffle = kwargs.pop('batchShuffle', True)
        self.eval_size = kwargs.pop('eval_size', 0.1)
        self.active_perceptrons = 0
        self.active_nodes = 0

        # Starting filters for each perceptron:
        self.nodes_partition = [np.sum(list_boost_filters[:i]) for i in range(len(list_boost_filters)+1)]
        print self.nodes_partition
        # Not really useful...
        # self.filters_per_perceptron = list_boost_filters[0]
        # self.filters_last_perceptron = list_boost_filters[-1]


        # Checks:
        if "train_split" in kwargs:
            raise ValueError('The option train_split is not used. Use eval_size instead.')


        # - Find total number of filters
        # - modify structure NN (no longer use separate)
        # - output shape for GT


        customBatchIterator = BatchIterator_Greedy(
            batch_size=self.batch_size,
            shuffle=self.batchShuffle,
        )

        # -------------------------------------
        # CONSTRUCT NETWORK:
        # -------------------------------------
        tot_num_filters = np.sum(list_boost_filters)
        self.layer_type = layers_info[name_trained_layer]['type']
        netLayers = deepcopy(fixed_input_layers)
        if self.layer_type=="conv":
            netLayers += [
                (layers.Conv2DLayer, {
                    'name': "greedyConv_1",
                    'num_filters': tot_num_filters,
                    'filter_size': self.filter_size1,
                    'pad':'same',
                    'W': Normal(std=1000),
                    'nonlinearity': rectify}),
                (MaskLayer,{
                    'name': 'mask',
                    'list_boost_filters': list_boost_filters,
                    'num_classes': self.num_classes
                    }),
                (layers.Conv2DLayer, {
                    'name': 'greedyConv_2',
                    'num_filters': self.num_classes,
                    'filter_size': self.filter_size2,
                    'pad':'same',
                    'W': Normal(std=1000),
                    'nonlinearity': segm_utils.softmax_segm})]

        elif self.layer_type=="trans_conv":
            netLayers += [
                (layers.TrasposedConv2DLayer, {
                    'name': "greedyConv_1",
                    'num_filters': tot_num_filters,
                    'filter_size': self.filter_size1,
                    'crop': trained_layer_args['crop'],  ## check if crop is always there
                    # 'W': Normal(std=1000),
                    'nonlinearity': rectify}),
                (MaskLayer,{
                    'name': 'mask',
                    'list_boost_filters': list_boost_filters,
                    'num_classes': self.num_classes
                    }),
                (layers.TrasposedConv2DLayer, {
                    'name': 'greedyConv_2',
                    'num_filters': self.num_classes,
                    'filter_size': self.filter_size2,
                    'crop': trained_layer_args['crop'],  ## check if crop is always there
                    # 'W': Normal(std=1000),
                    'nonlinearity': segm_utils.softmax_segm})]

        self.net = segmNeuralNet(
            layers=netLayers,
            batch_iterator_train = customBatchIterator,
            batch_iterator_test = customBatchIterator,
            objective_loss_function = segm_utils.categorical_crossentropy_segm,
            scores_train = [('trn pixelAcc', segm_utils.pixel_accuracy)],
            # scores_valid = [('val pixelAcc', pixel_accuracy)],
            y_tensor_type = T.ltensor3,
            eval_size=self.eval_size,
            regression = False,
            **kwargs
        )

        self.net._output_layer = self.net.initialize_layers()


        # Set all fixed layers as not trainable & reg:
        fixed_layers_names = [layer[1]['name'] for layer in fixed_input_layers]
        print fixed_layers_names
        for name in fixed_layers_names:
            if layers_info[name]['type']=='conv' or layers_info[name]['type']=='trans_conv':
                self.net.layers_[name].params[self.net.layers_[name].W].remove('trainable')
                self.net.layers_[name].params[self.net.layers_[name].b].remove('trainable')
                self.net.layers_[name].params[self.net.layers_[name].W].remove('regularizable')


        # Set the first greedy-layer as not trainable:
        self.net.layers_['greedyConv_1'].params[self.net.layers_['greedyConv_1'].W].remove('trainable')
        self.net.layers_['greedyConv_1'].params[self.net.layers_['greedyConv_1'].b].remove('trainable')

        # Set both greedy-layers as not regularizable:
        self.net.layers_['greedyConv_1'].params[self.net.layers_['greedyConv_1'].W].remove('regularizable')
        self.net.layers_['greedyConv_2'].params[self.net.layers_['greedyConv_2'].W].remove('regularizable')
        # tick = time.time()
        self.net.initialize()
        # tock = time.time()
        # print "Done! (%f sec.)\n\n\n" %(tock-tick)

        # -------------------------------------
        # SAVE INFO NET:
        # -------------------------------------
        # info['num_classes'] = self.num_classes
        # info.pop('update', None)
        # info.pop('on_epoch_finished', None)
        # info.pop('on_batch_finished', None)
        # info.pop('on_training_finished', None)
        # info.pop('noReg_loss', None)
        # for key in [key for key in info if 'update_' in key]:
        #     info[key] = info[key].get_value().item()
        # json.dump(info, file(info['logs_path']+'/info-net.txt', 'w'))



    def insert_weights(self, boostedPerceptron):
        '''
        In order the following operations are done:
         - Update mask main part: activate another node
         - Copy the 'new_node' weights in the main part of the net
         - Copy the boostedPerceptron weights in the 'new_node' part
         - Recompile the net

        Structure of parameters:
         - W1: (num_classes*num_filters1*num_nodes, num_inputs, filter_length1)
         - b1: (num_classes*num_filters1*num_nodes, )
         - W2: (num_classes, num_classes*num_filters1*num_nodes, filter_length2)
         - b2: (num_classes,)
        '''

        # ------------------
        # Update mask:
        # ------------------
        self.net.layers_['mask'].add_perceptron()
        self.active_perceptrons = self.net.layers_['mask'].active_perceptrons

        # ------------------
        # Get weights:
        # ------------------
        W1, b1, maskParam, W2, b2 = layers.get_all_param_values(self.net.layers_['greedyConv_2'])
        perc_W1, perc_b1, perc_W2, perc_b2 = layers.get_all_param_values(boostedPerceptron.net.layers_['greedyConv_2'])

        # --------------------
        # Update main part:
        # --------------------
        start = self.nodes_partition[self.active_perceptrons-1]
        stop = self.nodes_partition[self.active_perceptrons]
        slice_weights = slice(start,stop)
        # !!! For the moment I don't touch b2... !!! #
        b1[slice_weights] = perc_b1
        if self.layer_type=="conv":
            W1[slice_weights,:,:] = perc_W1
            W2[:,slice_weights,:] = perc_W2
        if self.layer_type=="trans_conv":
            W1[:,slice_weights,:] = perc_W1
            W2[slice_weights,:,:] = perc_W2
        layers.set_all_param_values(self.net.layers_['greedyConv_2'], [W1, b1, maskParam, W2, b2])


class MaskLayer(layers.Layer):
    '''
    --------------------------
    Subclass of lasagne.layers.Layer:
    --------------------------

    The received input should be in the form: (N, num_classes*num_nodes, dim_x, dim_y)

    Inputs:
     - num_filters1 (5)
     - num_classes (1)

    The only parameter of the layer is a 2-dim array containing the slice extremes
    deciding the active nodes. When initialized, no nodes are active.
    '''

    def __init__(self, incoming, *args, **kwargs):
        self.num_classes = kwargs.pop('num_classes', 2)
        self.list_boost_filters = kwargs.pop('list_boost_filters', None)
        if self.list_boost_filters is None:
            raise ValueError("List of filters not passed for boosting training")

        super(MaskLayer, self).__init__(incoming, *args, **kwargs)
        self.active_perpectrons = 0
        self.active_nodes = self.add_param(np.ones(1, dtype=np.int8), (1,), name='active_nodes', trainable=False, regularizable=False)
        #
        self.active_nodes.set_value([0])


    def add_perceptron(self):
        '''
        Add one node.

        The first time nothing is done (just add new node, but still no active
            ones in the main net)
        '''
        self.active_perpectrons += 1
        self.active_nodes.set_value([self.list_boost_filters[:self.active_perpectrons].sum()])


    def get_output_for(self, input, **kwargs):
        # Set to zero all the not-active perceptrons: (avoid backprop)
        return T.set_subtensor(input[:,self.active_nodes[0]:,:,:], 0.)



