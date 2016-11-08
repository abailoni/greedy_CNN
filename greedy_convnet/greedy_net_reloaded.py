__author__ = "abailoni"

'''
New version of greedyNet, detecting automatically which layers to train and in
which order, given the structure of a net.
'''

import time
from copy import deepcopy, copy
from collections import OrderedDict
import numpy as np
from warnings import warn

import theano.tensor as T
import lasagne.layers as layers
from lasagne.layers import set_all_param_values, get_all_param_values, get_output_shape
from lasagne.nonlinearities import rectify


import pretr_nets.vgg16 as vgg16

from mod_nolearn.segm import segmNeuralNet
import mod_nolearn.segm.segm_utils as segm_utils
from greedy_convnet.sub_nets import greedyLayer_reload as greedyLayer

from greedy_convnet.sub_nets import boostedPerceptron

import various.utils as utils


def restore_greedyModel(model_name, path_logs='./logs/'):
    warn("This way to restore a model has been deprecated, please use the method load_pretrained_layers() instead and load the weights.")
    return utils.restore_model(path_logs+model_name+'/model.pickle')


class greedyNet(object):
    '''
    Example of network:

            Input
            /   \
           /     \
        conv1   conv2
           \     /
            \   /
            conv3
              |
            Output

    The detected input-output ways are:

    [
        ['input', 'conv1', 'conv3', 'out'],
        ['input', 'conv2', 'conv3', 'out']
    ]

    Starting from the first way, 'conv1' is trained. Then we try to
    train 'conv3', but 'conv2' is in the requirements and it has not been
    trained.
    Going then to the next way, 'conv2' is trained. Finally we can also
    train 'conv3'. The output layer is reached, thus the process is stopped.
    '''

    def __init__(self, nolearn_layers, **FULLNET_kwargs):
        '''
        It takes the structure of a network (in the form of a nolearn dictionary)
        and handle the greedy training, deciding in which order the layers
        should be trained.

        REMARK:
        for the moment it works only with the standard dictionary of nolearn

        The output layer should be the last one of the passed list.

        Options and parameters:

         - BASE_PATH_LOG
         - model_name

        All options that hold in general for all nolearn-nets, like:
            - batch_size
            - eval_size
            - num_classes
            - batchShuffle
            - ...
        '''
        self.input_layers = nolearn_layers

        # Temp-network used only for computing output spatial dim and
        # trained layers:
        self.whole_net = segmNeuralNet(layers=nolearn_layers)
        self.whole_net.initialize_layers()

        # Compute output spatial dimensions:
        self.layers_info = OrderedDict()
        for layer_name in self.whole_net.layers_:
            self.layers_info[layer_name] = {}
        self._compute_all_spatial_sizes()

        # -------------------------------
        # Detect layers informations:
        # -------------------------------
        self.names = []
        for i, layer in enumerate(self.input_layers):
            # ---------------------------------
            # Detect or create name for layer:
            # ---------------------------------
            if isinstance(layer[1], dict):
                # Newer format: (Layer, {'layer': 'kwargs'})
                layer_factory, layer_kw = layer
                layer_kw = layer_kw.copy()
            else:
                # The legacy format: ('name', Layer)
                layer_name, layer_factory = layer
                layer_kw = {'name': layer_name}

            if 'name' not in layer_kw:
                layer_kw['name'] = self._layer_name(layer_factory, i)
                self.input_layers[i][1]['name'] = layer_kw['name']

            name = layer_kw['name']
            self.names.append(name)
            layer_dict = {}
            self.layers_info[name] = layer_dict

            # ---------------------------------
            # Check if it can be trained greedily:
            # ---------------------------------
            layer_dict['trained_greedily'] = False
            layer_dict['trained'] = False
            if layer_factory==layers.Conv2DLayer:
                layer_dict['trained_greedily'] = True
                layer_dict['type'] = "conv"
            elif layer_factory==layers.TransposedConv2DLayer:
                layer_dict['trained_greedily'] = True
                layer_dict['type'] = "trans_conv"
            else:
                params = self.whole_net.layers_[name].get_params()
                if len(params)!=0:
                    raise NotImplemented("A greedy training of the layer %s is not implemented!")
                layer_dict['type'] = None # other layer-type

            # ---------------------------------
            # Check the requirements of the layer
            # (i.e. all the layers that are inputed to the layer):
            # ---------------------------------
            if layer_factory==layers.InputLayer:
                layer_dict['req'] = False
                layer_dict['trained'] = True
            else:
                if 'incoming' in layer_kw:
                    layer_dict['req'] = [layer_kw['incoming']]
                elif 'incomings' in layer_kw:
                    layer_dict['req'] = layer_kw['incomings']
                else:
                    layer_dict['req'] = [self.names[i-1]]


        # -------------------------------
        # Contruct all possible ways from input to output:
        # -------------------------------
        ways_to_output = [[self.names[-1]]]

        self.ways_to_output = self._detect_ways_to_input(ways_to_output)
        self.num_ways = len(self.ways_to_output)
        self.selected_way = 0


        # -------------------------------
        # Other passed arguments:
        # -------------------------------
        self.greedy_layer_class = greedyLayer
        self.boosted_perceptron_class = boostedPerceptron
        self.model_name = FULLNET_kwargs.pop('model_name', 'greedyNET')
        self.BASE_PATH_LOG = FULLNET_kwargs.pop('BASE_PATH_LOG', "./logs/")
        self.BASE_PATH_LOG_MODEL = self.BASE_PATH_LOG+self.model_name+'/'

        self.subNets = {}
        self.FULLNET_kwargs = copy(FULLNET_kwargs)
        self.trained_weights = {}




    def _detect_ways_to_input(self, ways_to_output, idx_way=0):
        '''
        Construct all the ways from the input layer to the output one.

        The output is a list of layer's names, e.g.
        [
            ['input', 'conv1', 'conv3'],
            ['input', 'conv2', 'conv3']
        ]
        '''
        current_way = ways_to_output[idx_way]
        layer = current_way[0]

        # Loop until we reach the input:
        while self.layers_info[layer]['req']:
            if len(self.layers_info[layer]['req'])==1:
                # Just one requirements:
                current_way.insert(0,self.layers_info[layer]['req'][0])
            else:
                # More requirements, create other ways and go for recursion:
                idx_new_ways = [idx_way]
                for i in range(len(self.layers_info[layer]['req'])-1):
                    ways_to_output.append(deepcopy(ways_to_output[idx_way]))
                    idx_new_ways.append(len(ways_to_output)-1)

                for new_way, req_layer in zip(idx_new_ways, self.layers_info[layer]['req']):
                    ways_to_output[new_way].insert(0, req_layer)
                    ways_to_output = self._detect_ways_to_input(ways_to_output, new_way)
                break
            layer = current_way[0]

        return ways_to_output


    def _layer_name(self, layer_class, index):
        '''
        Helper method to create name (taken from nolearn)
        '''
        return "{}{}".format(
            layer_class.__name__.lower().replace("layer", ""), index)


    def perform_next_greedy_step(
            self,
            fit_perceptrons,
            finetune,
            kwargs_perceptron=None,
            kwargs_finetune=None,
            **kwargs
        ):

        if not kwargs_finetune:
            kwargs_finetune = {}
        if not kwargs_perceptron:
            kwargs_perceptron = {}

        # Get informations about next greedy step:
        fixed_input_layers, trained_layer = self._get_next_step()

        assert trained_layer!=False, "All the network has been trained. No other greedy steps to perform."

        # Detect informations about the layer to be trained:
        boost_filters = kwargs.pop('boost_filters', 10)
        list_boost_filters = self._init_boosting(trained_layer, boost_filters)

        # ----------------------------
        # Start boosting training:
        # ----------------------------

        # Compile greedy layer:
        num_pretrained_perc = self.init_greedyLayer(trained_layer, fixed_input_layers, list_boost_filters, kwargs_finetune)
        print "Num. pretrained perceptrons: %d" %num_pretrained_perc

        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # Training of the first node:
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        if num_pretrained_perc==0:
            # Compile first node:
            self.init_boostedPerceptron(trained_layer, fixed_input_layers, list_boost_filters[0], kwargs_perceptron)

            # Train first node: (and backup node)
            self.train_boostedPerceptron(trained_layer, fit_perceptrons)

            # Insert in layer: (and backup layer)
            self.update_weights_greedyLayer(trained_layer)
            self.pickle_greedyLayer_weights(trained_layer)

        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # Training of other nodes:
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        for num_filters in list_boost_filters[num_pretrained_perc+1:]:

            # Compile new boostedNode:
            self.init_boostedPerceptron(trained_layer, fixed_input_layers, num_filters, kwargs_perceptron)

            # Train and insert weights:
            self.train_boostedPerceptron(trained_layer, fit_perceptrons)
            self.update_weights_greedyLayer(trained_layer)
            self.pickle_greedyLayer_weights(trained_layer)


            # Finetune active nodes in greedyLayer:
            self.finetune_greedyLayer(trained_layer, finetune)
            self.pickle_greedyLayer_weights(trained_layer)

        # ----------------------------
        # Set new layer and all the following
        # not-greedily-trainable as trained:
        # ----------------------------
        self._insert_new_layer(trained_layer)
        self._check_trained_layers()




    def update_weights_greedyLayer(self, trained_layer):
        perceptron_name = "perceptron_"+trained_layer
        self.subNets[trained_layer].insert_weights(self.subNets[perceptron_name])


    def _get_next_step(self):
        '''
        Returns the sequence of fixed layers necessary to train the next
        greedy step and the name of the trained layer.
        '''
        way_num = self.selected_way
        current_way = self.ways_to_output[way_num]
        info = self.layers_info
        layer_to_train = False
        for i, layer_name in enumerate(current_way):
            if info[layer_name]['trained_greedily']:
                if not info[layer_name]['trained']:
                    layer_to_train = layer_name

                    # --------------------------------------------
                    # Check if all the requirements are satisfied
                    # otherwise go to next way:
                    # --------------------------------------------
                    for req in info[layer_to_train]['req']:
                        if not info[req]['trained']:
                            self.selected_way = (self.selected_way+1)%self.num_ways
                            return self._get_next_step()

                    num_inputs = i
                    break

        # All the network has been trained:
        if not layer_to_train:
            print "All the network has been trained!"
            return self.input_layers, False

        # --------------------------------------------
        # Collect fixed inputs for the greedy step:
        # --------------------------------------------
        fixed_input = []
        for i in range(num_inputs):
            name_layer = current_way[i]
            idx_layer = self.names.index(name_layer)
            fixed_input.append(self.input_layers[idx_layer])
        return fixed_input, layer_to_train



    def _init_boosting(self, trained_layer, boost_filters):
        '''
        Compute how to boosting-train the filters in the selected trained_layer.
        (i.e. how the perceptrons are defined and with how many filters)

        Inputs:
            - trained_layer: name of the layer to train
            - boost_filters: number of filters per perceptron

        Returns an array a s.t.:
            - a.shape[0] = number of perceptrons to train
            - a[i] = num. of filters in perceptron i
        '''
        # Check if spatial size has been computed:
        if "output_shape" not in self.layers_info[trained_layer]:
            self._compute_all_spatial_sizes()

        # Get num filters layer:
        out_layer = self.whole_net.layers_[trained_layer]
        if hasattr(out_layer, 'num_filters'):
            num_filters = out_layer.num_filters
        else:
            raise ValueError("The greeedy-trained layer do not have a defined number of filters")

        list_boost_filters = [boost_filters]*(int(num_filters)/int(boost_filters))
        list_boost_filters += [int(num_filters)%int(boost_filters)]

        return np.array(list_boost_filters)


    def _compute_all_spatial_sizes(self):
        '''
        Save the output shapes of all layers.
        '''
        for layer_name in self.whole_net.layers_:
            self.layers_info[layer_name]["output_shape"] = get_output_shape(self.whole_net.layers_[layer_name])


    def init_greedyLayer(self, trained_layer, fixed_input_layers, list_boost_filters, kwargs_finetune):
        '''
        Initialize the network that will finetune the pre-trained perceptrons
        of the given trained_layer.
        '''
        # If not preLoaded, initialize new network:
        if trained_layer not in self.subNets:
            logs_path = self.BASE_PATH_LOG_MODEL+trained_layer+'/'
            utils.create_dir(logs_path)
            params = deepcopy(kwargs_finetune)
            params['log_frequency'] = 1
            params['log_filename'] = 'log.txt'
            params['subLog_filename'] = 'sub_log.txt'
            # params['pickleModel_mode'] = 'on_epoch_finished'
            # params['trackWeights_pdfName'] = 'weights.txt'
            params['pickle_filename'] = 'model.pickle'
            params['logs_path'] = logs_path

            idx_layer = self.names.index(trained_layer)
            layer_kargs = self.input_layers[idx_layer][1]

            self.subNets[trained_layer] = self.greedy_layer_class(
                fixed_input_layers,
                self.layers_info,
                trained_layer,
                layer_kargs,
                self.FULLNET_kwargs,
                list_boost_filters,
                self.trained_weights,
                **params
            )
            return 0
        else:
            raise NotImplemented("Loading greedy layer from weights not implemented yet")
            # Stuff to do:
            #
            # - Return pre-trained percetrons
            # - init and load all perceptrons weights in greedyLayer (not really feasible with the current implementation that saves only the last learned perc.)
            # - init and load the full pretrained greedyLayer
            # - (update active_perceptrons)
            pass


    def init_boostedPerceptron(self, trained_layer, fixed_input_layers, filters_in_perceptron, kwargs_boostedPerceptron, num_perceptron=False):
        '''
        Initialize the network that will train one perceptron.

        Remark: at the moment only the last trained perceptron is kept in memory, not all of them.
        '''


        perceptron_name = "perceptron_"+trained_layer

        params = deepcopy(kwargs_boostedPerceptron)
        logs_path = self.BASE_PATH_LOG_MODEL+perceptron_name+'/'
        utils.create_dir(logs_path)
        params['log_frequency'] = 1
        params['subLog_filename'] = 'sub_log.txt'
        params['log_filename'] = 'log.txt'
        # params['pickleModel_mode'] = 'on_epoch_finished'
        # params['trackWeights_pdfName'] = 'weights.txt'
        params['pickle_filename'] = 'model.pickle'
        params['logs_path'] = logs_path


        self.subNets[perceptron_name] = self.boosted_perceptron_class(
            fixed_input_layers,
            self.layers_info,
            trained_layer,
            filters_in_perceptron,
            self.subNets[trained_layer],
            self.trained_weights,
            **params
        )


    def train_boostedPerceptron(self, trained_layer, fit_routine):
        # Train perceptron:
        num_perceptron  = self.subNets[trained_layer].active_perceptrons
        perceptron_name = "perceptron_"+trained_layer
        print "\n\n----------------------------------------------------"
        print "Training Boosted Perceptron %d - %s" %(num_perceptron, trained_layer)
        print "----------------------------------------------------\n\n"
        self.subNets[perceptron_name].net = fit_routine(self.subNets[perceptron_name].net,num_perceptron=num_perceptron)



    def finetune_greedyLayer(self, trained_layer, finetune_routine):
        # Finetune:
        num_perceptron  = self.subNets[trained_layer].active_perceptrons
        print "\n\n--------------------------------------------------------"
        print "Finetuning layer - %s, %d active perceptrons " %(trained_layer, num_perceptron)
        print "--------------------------------------------------------\n\n"
        self.subNets[trained_layer].net = finetune_routine(self.subNets[trained_layer].net,num_perceptron=num_perceptron)


    def pickle_greedyLayer_weights(self, trained_layer):
        '''
        Collect and pickle weights of the greedy layer (including
            the last additional classification layer)

        PROBLEM to solve: no distinction before/after finetuning....
        '''
        # Collect weights:
        weights = self.subNets[trained_layer].get_greedy_weights()

        # Pickle weights:
        active_perceptrons = self.subNets[trained_layer].active_perceptrons
        utils.create_dir(self.BASE_PATH_LOG_MODEL+"greedy_weights/%s/" %(trained_layer) )
        utils.pickle_model(weights, self.BASE_PATH_LOG_MODEL+"greedy_weights/%s/finetuning_%d_perceptr.pkl" %(trained_layer, active_perceptrons) )


    def load_pretrained_layers(self, pretrained_greedy_net=None):
        '''
        Load the weights of some previously pretrained layers.
        Set all of them as already trained.
        '''
        model = pretrained_greedy_net if pretrained_greedy_net else self

        filename = model.BASE_PATH_LOG_MODEL+"weights_trainedLayers.pkl"
        if utils.check_file(filename):
            # if self.verbose:
            #     print "Loading pretrained weights"
            self.trained_weights = utils.restore_model(filename)

            # Set them as trained:
            for layer in self.trained_weights:
                self.layers_info[layer]['trained'] = True

            # Set the not-greedily-trained as trained:
            self._check_trained_layers()
        else:
            print "No pretrained layers found"



    def _insert_new_layer(self, trained_layer):
        '''
        It labels trained_layer as trained, saves learned weights,
        updates the main-weights-dictionary and pickles it.
        '''

        # Collect weights:
        net = self.subNets[trained_layer].net
        params_trained_layer = net.layers_['greedyConv_1'].get_params()
        params_values = [param.get_value() for param in params_trained_layer]
        self.trained_weights[trained_layer] = params_values

        # Set the layer as trained:
        self.layers_info[trained_layer]['trained'] = True


        # Pickle all trained weights:
        utils.pickle_model(self.trained_weights, self.BASE_PATH_LOG_MODEL+"weights_trainedLayers.pkl")




    def _check_trained_layers(self):
        '''
        After training one layer, it labels all the following layers that
        are not greedily-trained as 'trained'.
        '''
        # Select the current way:
        way_num = self.selected_way
        current_way = self.ways_to_output[way_num]
        info = self.layers_info

        # Check consistency labels 'trained':
        for i, layer_name in enumerate(current_way):
            if not info[layer_name]['trained']:
                # If not-trained and trained greedily, stop:
                if info[layer_name]['trained_greedily']:
                    break

                trained = True
                for req in info[layer_name]['req']:
                    # If at least one of the requirements is not trained,
                    # leave the layer as not-trained:
                    if not info[req]['trained']:
                        trained = False
                        break
                info[layer_name]['trained'] = trained

                # If not trained, all the following layers won't be 'trained':
                if not trained:
                    break


    '''

    ####################################
    ## OBSOLETE: need to be updated with the new pickling method
    ####################################

    #------------------------
    # UTILS METHODS:
    #------------------------
    To import previously pretrained full or partial Greedy models:

      - create a new GreedyNET or import existing one using
        restore_greedyModel()

      - in order not to overwrite the old model, call update_all_paths()
        to update the name (and change the logs path)

      - if subNets should be loaded, call load_subNets()
    '''

    def update_all_paths(self, newname_model, new_path=None):
        '''
        After restoring a greedy model or importing pre-trained subNets, this
        method should be called to avoid inconsistencies in logs and saved data.

        In order what it does:
          - update name main model
          - update main paths (and copy folders)
          - update all paths of all subNets
        '''
        warn("This mehtod is obsolete and needs to be updated")


        old_path_model = self.BASE_PATH_LOG_MODEL
        if new_path:
            self.BASE_PATH_LOG = new_path
        self.model_name = newname_model
        self.BASE_PATH_LOG_MODEL = self.BASE_PATH_LOG+self.model_name+'/'

        # Copy directories and logs:
        import mod_nolearn.utils as utils
        utils.copyDirectory(old_path_model, self.BASE_PATH_LOG_MODEL)

        # Update paths:
        self.net.update_logs_path(self.BASE_PATH_LOG_MODEL)
        self._update_subNets_paths()


    def load_subNets(self, preLoad):
        '''
        Input:
            - preLoad: a dictionary such that:
                - the keys are the names of the nodes (e.g. 'cnv_L0_G0')
                - each element contains a tuple such that:
                     (path_to_pretr_model, train_flag, nodes_trained)
                     ('logs/model_A/', True, nodes_trained)

        The third options indicates the pretrained nodes only in the case of a greedyLayer.

        In particular we have:
                train = True if active_nodes>load[2] else False
        where the active_nodes are the one in the MAIN part of the net, w/o considering the new node.
        Conclusion: just put how many nodes have been aleardy trained in the MAIN part of the net (given by convSoft.active_nodes)

        What it does in order:
            - import subNets
            - copy folders/logs in main model (and delete previous versions)
            - update all paths of subNets
        '''
        warn("This mehtod is obsolete and needs to be updated.")
        self.preLoad = preLoad
        for net_name in preLoad:
            load = preLoad[net_name]
            if load[0]:
                # Imported from other pretrained model:
                subNet_old_path = load[0]+net_name+'/'
                subNet_new_path = self.BASE_PATH_LOG_MODEL+net_name+'/'
                if 'reg' in net_name:
                    self.regr[net_name] = utils.restore_model(subNet_old_path+'routine.pickle')
                elif 'cnv' in net_name:
                    self.convSoftmax[net_name] = utils.restore_model(subNet_old_path+'routine.pickle')
                else:
                    raise ValueError("Not recognized netname")
                # Delete possible previous folders:
                utils.deleteDirectory(subNet_new_path[:-1])
                # Copy old subNet:
                # utils.create_dir(subNet_new_path)
                utils.copyDirectory(subNet_old_path, subNet_new_path)

        self._update_subNets_paths()


    def _update_subNets_paths(self):
        for net_name in self.regr:
            self.regr[net_name].net.update_logs_path(self.BASE_PATH_LOG_MODEL+net_name+'/')
        for net_name in self.convSoftmax:
            self.convSoftmax[net_name].net.update_logs_path(self.BASE_PATH_LOG_MODEL+net_name+'/')


    def pickle_greedyNET(self):
        utils.pickle_model(self, self.BASE_PATH_LOG_MODEL+'model.pickle')


# class greedyProcess(object):
#     '''
#     It takes the structure of a network (in the form of a nolearn dictionary)
#     and handle the greedy training, deciding in which order the layers
#     should be trained.
#     '''


#     def __init__(self, nolearn_layers):
#         '''
#         The output layer should be the last one of the list.

#         '''
#         self.input_layers = nolearn_layers

#         # Detect layers informations:
#         self.layers_info = OrderedDict()
#         for i, layer in enumerate(self.input_layers):
#             if isinstance(layer[1], dict):
#                 # Newer format: (Layer, {'layer': 'kwargs'})
#                 layer_factory, layer_kw = layer
#                 layer_kw = layer_kw.copy()
#             else:
#                 # The legacy format: ('name', Layer)
#                 layer_name, layer_factory = layer
#                 layer_kw = {'name': layer_name}

#             if 'name' not in layer_kw:
#                 layer_kw['name'] = self._layer_name(layer_factory, i)

#             layer_dict = {}
#             self.layers_info[name] = layer_dict
#             # Used only in few situations: (see requirements)
#             layer_dict['name'] = name

#             # Check if can be trained greedily:
#             layer_dict['trained_greedily'] = False
#             if isinstance(layer_factory, (layers.Conv2DLayer, layers.TransposedConv2DLayer)):
#                 layer_dict['trained_greedily'] = True

#             # Check the immediate requirements of the layer:
#             if isinstance(layer_factory, layers.InputLayer):
#                 layer_dict['req'] = False
#             else:
#                 if 'incoming' in layer_kw:
#                     layer_dict['req'] = [layer_kw['incoming']]
#                 elif 'incomings' in layer_kw:
#                     layer_dict['req'] = layer_kw['incomings']
#                 else:
#                     layer_dict['req'] = [self.layers_info[i-1]['name']]

#         # Contruct all possible ways from output to input:
#         ways_to_input = [[self.layers_info[-1]['name']]]
#         ways_to_input = self._detect_way_to_input(ways_to_input)



#     def _detect_way_to_input(self, ways_to_input, idx_way=0):
#         current_way = ways_to_input[idx_way]
#         layer = current_way[-1]
#         while self.layers_info[layer]['req']:
#             if len(self.layers_info[layer]['req'])==1:
#                 # Just one requirements:
#                 current_way.insert(0,self.layers_info[layer]['req'][0])
#             else:
#                 # More requirements, create other ways:
#                 for i in range(len(self.layers_info[layer]['req'])-1):
#                     ways_to_input.append(copy(ways_to_input[idx_way]))

#                 for i, req_layer in enumerate(self.layers_info[layer]['req']):
#                     ways_to_input[idx_way+i].insert(0, req_layer)
#                     ways_to_input = self._detect_way_to_input(ways_to_input, idx_way+i)
#                 break

#             layer = current_way[-1]

#         return ways_to_input



#     def _layer_name(self, layer_class, index):
#         '''
#         Taken from nolearn library.
#         '''
#         return "{}{}".format(
#             layer_class.__name__.lower().replace("layer", ""), index)




# netLayers = [
#     # layer dealing with the input data
#     (layers.InputLayer, {
#         'name': 'inputLayer',
#         'shape': (None, None, None, None)}),
#     (layers.Conv2DLayer, {
#         'name': 'conv1',
#         'num_filters': 5,
#         'filter_size': 3,
#         'nonlinearity': rectify}),
#     (layers.Conv2DLayer, {
#         'name': 'conv2',
#         'num_filters': 2,
#         'filter_size': 3,
#         'pad':'same',
#         'nonlinearity': identity}),
#     # New node:
#     (layers.Conv2DLayer, {
#         'incoming': 'inputLayer',
#         'name': 'conv1_newNode',
#         'num_filters': 5,
#         'filter_size': 3,
#         'pad':'same',
#         'nonlinearity': rectify}),
#     (layers.Conv2DLayer, {
#         'name': 'conv2_newNode',
#         'num_filters': 2,
#         'filter_size': 3,
#         'pad':'same',
#         'nonlinearity': identity}),
#     (boosting_mergeLayer, {
#         'incomings': ['conv2', 'conv2_newNode'],
#         'merge_function': T.add,
#         'name': 'boosting_merge'}),
#     (layers.NonlinearityLayer,{
#         'name': 'final_nonlinearity',
#         'incoming': 'boosting_merge',
#         'nonlinearity': segm_utils.softmax_segm}),
# ]





