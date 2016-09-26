import time
from copy import deepcopy, copy
from collections import OrderedDict


import theano.tensor as T
import lasagne.layers as layers
from lasagne.layers import set_all_param_values, get_all_param_values
from lasagne.nonlinearities import rectify


import pretr_nets.vgg16 as vgg16

from mod_nolearn.segm import segmNeuralNet
import mod_nolearn.segm.segm_utils as segm_utils

from greedy_convnet.sub_nets import greedyLayer, greedyLayer_ReLU, boostedNode, boostedNode_ReLU

import various.utils as utils


def restore_greedyModel(model_name, path_logs='./logs/'):
    # After this, if the model is renamed, the method update_all_paths should be called
    return utils.restore_model(path_logs+model_name+'/model.pickle')

class greedyNet(object):

    def __init__(self, nolearn_layers, **kwargs):
        '''
        It takes the structure of a network (in the form of a nolearn dictionary)
        and handle the greedy training, deciding in which order the layers
        should be trained.

        The output layer should be the last one of the passed list.

        Options and parameters:

         - BASE_PATH_LOG
         - model_name
        '''
        self.input_layers = nolearn_layers

        # -------------------------------
        # Detect layers informations:
        # -------------------------------
        self.layers_info = OrderedDict()
        self.names = []
        for i, layer in enumerate(self.input_layers):
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

            name = layer_kw['name']
            self.names.append(name)
            layer_dict = {}
            self.layers_info[name] = layer_dict

            # Check if can be trained greedily:
            layer_dict['trained_greedily'] = False
            layer_dict['trained'] = False
            if layer_factory==layers.Conv2DLayer: ##### MOD!!! #####
                layer_dict['trained_greedily'] = True


            # Check the immediate requirements of the layer:
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
        self.selected_way, self.num_ways = 0, len(self.ways_to_output)


        # -------------------------------
        # Other passed arguments:
        # -------------------------------
        self.greedy_layer_class = greedyLayer_ReLU
        self.boosted_node_class = boostedNode_ReLU
        self.model_name = kwargs.pop('model_name', 'greedyNET')
        self.BASE_PATH_LOG = kwargs.pop('BASE_PATH_LOG', "./logs/")
        self.BASE_PATH_LOG_MODEL = self.BASE_PATH_LOG+self.model_name+'/'




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
                # More requirements, create other ways and go recursive:
                for i in range(len(self.layers_info[layer]['req'])-1):
                    ways_to_output.append(copy(ways_to_output[idx_way]))
                for i, req_layer in enumerate(self.layers_info[layer]['req']):
                    ways_to_output[idx_way+i].insert(0, req_layer)
                    ways_to_output = self._detect_ways_to_input(ways_to_output, idx_way+i)
                break
            layer = current_way[0]

        return ways_to_output



    def _layer_name(self, layer_class, index):
        return "{}{}".format(
            layer_class.__name__.lower().replace("layer", ""), index)


    def perform_next_greedy_step(
            self,
            fit_nodes,
            finetune,
            kwargs_nodes=None,
            kwargs_finetune=None,
            **kwargs
        ):
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

        The detected ways are:

        [
            ['input', 'conv1', 'conv3', 'out'],
            ['input', 'conv2', 'conv3', 'out']
        ]

        Starting from the first way, 'conv1' will be trained. Then we try to
        train 'conv3', but 'conv2' is in the requirements and it has not been
        trained.
        Going then to the next way, 'conv2' is trained. Finally we can also
        train 'conv3'. The output layer is reached, thus the process is stopped.
        '''
        if not kwargs_finetune:
            kwargs_finetune = {}
        if not kwargs_nodes:
            kwargs_nodes = {}

        # Get informations aboout next greedy step:
        fixed_input_layers, trained_layer = self._get_next_step()

        # Detect informations about the layer to be trained:
        boost_filters = kwargs.pop('boost_filters', 10)
        list_boost_filters = self._init_boosting(trained_layer, boost_filters)

        # ----------------------------
        # Start boosting training:
        # ----------------------------

        # Compile greedy layer:
        num_pretrained_nodes = self.init_greedyLayer(trained_layer, fixed_input_layers, list_boost_filters, kwargs_finetune)


        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # Training of the first node:
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        if num_pretrained_nodes==0:
            # Compile first node:
            self.init_node(trained_layer, fixed_input_layers, kwargs_nodes, filters=list_boost_filters[0], node=0)

            # Train first node: (and backup node)
            self.train_node(trained_layer, fit_nodes, node=0)

            # Insert in layer: (and backup layer)
            self.insert_weights(trained_layer, node=0)

        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        # Training of the other nodes:
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        for num_node, num_filters in enumerate(list_boost_filters[num_pretrained_nodes+1:]):

            # Compile new boostedNode:
            self.init_node(trained_layer, fixed_input_layers, kwargs_nodes, filters=num_filters, node=num_node)

            # Train and insert weights:
            self.train_node(trained_layer, fit_nodes, node=num_node)
            self.insert_weights(trained_layer, node=num_node)

            # Finetune active nodes in greedyLayer:
            self.finetune_greedyLayer(trained_layer, finetune, node=num_node)

        # ----------------------------
        # Set new layer and all the following not-greedily-trainable as trained:
        # ----------------------------
        # To be continued...



    def _get_next_step(self):
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
        Detect also size and adjust the GrTruth...
        '''


    # def __init__(self, num_VGG16_layers, mod=None, **kwargs):
    #     '''
    #     Initialize a network that uses just the first layers of VGG16.

    #     Inputs:
    #      - how many layers to keep of VGG16 (temporary max=2 due to MaxPooling layers)


    #     Options and parameters:

    #      - mod: the accepted options are 'basic' or 'ReLU'. With 'basic', the
    #         greedy network uses a softmax nonlinearity that is later transformed
    #         in a ReLU nonlinearity. With 'ReLU', the rectify linearity is used
    #         from the beginning (check boostedNode_ReLU)
    #      - eval_size: percentage between training and validation data for nolearn
    #         NeuralNet class
    #      - BASE_PATH_LOG
    #      - model_name
    #      - further modNeuralNet parameters


    #     '''
    #     # ------------------------------------
    #     # Compile network with vgg16 layers:
    #     # ------------------------------------
    #     self.num_layers = 0
    #     if mod=='basic':
    #         self.greedy_layer_class = greedyLayer
    #         self.boosted_node_class = boostedNode
    #     elif mod=='ReLU':
    #         self.greedy_layer_class = greedyLayer_ReLU
    #         self.boosted_node_class = boostedNode_ReLU
    #     else:
    #         raise ValueError("A mode is required. Accepted types: 'basic', 'ReLU'")

    #     self.num_VGG16_layers = int(num_VGG16_layers)
    #     self.eval_size =  kwargs.pop('eval_size',0.)
    #     self.model_name = kwargs.pop('model_name', 'greedyNET')
    #     self.BASE_PATH_LOG = kwargs.pop('BASE_PATH_LOG', "./logs/")


    #     self.BASE_PATH_LOG_MODEL = self.BASE_PATH_LOG+self.model_name+'/'
    #     self.layers = vgg16.nolearn_vgg16_layers()[:self.num_VGG16_layers+1]
    #     fixed_kwargs = {
    #         'objective_loss_function': segm_utils.categorical_crossentropy_segm,
    #         'y_tensor_type': T.ltensor3,
    #         'eval_size': self.eval_size,
    #         'regression': False,
    #         'logs_path': self.BASE_PATH_LOG_MODEL
    #     }
    #     self.net_kwargs = kwargs.copy()
    #     self.net_kwargs.update(fixed_kwargs)
    #     self.net = segmNeuralNet(
    #         layers=self.layers,
    #         scores_train=[('trn pixelAcc', segm_utils.pixel_accuracy)],
    #         scores_valid=[('val pixelAcc', segm_utils.pixel_accuracy)],
    #         **self.net_kwargs
    #     )

    #     # ---------------------
    #     # Initialize net:
    #     # ---------------------
    #     # print "Compiling greedy network..."
    #     # tick = time.time()
    #     self.net.initialize()
    #     # tock = time.time()
    #     # print "Done! (%f sec.)\n\n\n" %(tock-tick)


    #     # --------------------
    #     # Copy vgg16 weights:
    #     # --------------------
    #     self.net = vgg16.nolearn_insert_weights_vgg16(self.net,self.num_VGG16_layers)


    #     self.output_channels = self.net.layers[-1][1]['num_filters']
    #     self.last_layer_name = self.net.layers[-1][1]['name']
    #     self.regr, self.convSoftmax = {}, {}
    #     self.preLoad = {}


    def train_new_layer(self,
            (fit_boosted_nodes, num_nodes, kwargs_boosted_nodes),
            (finetune_nodes, kwargs_finetune_nodes),
            adjust_convSoftmax=None):
        '''
        It trains a new layer of the greedy network.

        Inputs:

            - tuple (fit_boosted_nodes, num_nodes, kwargs_boosted_nodes)
                where fit_boosted_nodes is a callable function of type fun(net)
                that trains one node and return the net
                (where net is an instance of nolearn NeuralNet)

            - tuple (finetune_nodes, kwargs_finetune_nodes)
            where finetune_nodes is a callable training and finetuning the computed nodes

            - adjust_convSoftmax is an optional callable that adjusts the
                parameters of the greedyLayer after the training of each boosted
                node (e.g. learning rate, etc...)
        '''


        # ------------------------------------------------
        # Loop for more subsets of nodes: (fixed to one for now)
        #
        # (to be improved with parallelization)
        # ------------------------------------------------
        num_subsets = 1
        nodes_subsets = []
        for indx_subset in range(num_subsets):
            # -------------------------------------------
            # Fit first node and initialize greedyLayer:
            # -------------------------------------------
            greedyLayer_name = "cnv_L%d_G%d"%(self.num_layers,indx_subset)
            self.init_greedyLayer(greedyLayer_name, kwargs_finetune_nodes, num_nodes)

            boostedNode_name = "regr_L%dG%dN%d" %(self.num_layers,indx_subset,0)
            self.init_boostedNode(boostedNode_name, kwargs_boosted_nodes, greedyLayer_name)
            train_flag = self.train_boostedNode(boostedNode_name,  fit_boosted_nodes, indx_subset, 0)

            # Check if first node was trained or it was loaded
            # but not inserted in the main greedyLayer:
            if train_flag or self.convSoftmax[greedyLayer_name].net.layers_['mask'].first_iteration:
                self.convSoftmax[greedyLayer_name].insert_weights(self.regr[boostedNode_name])

            # -----------------------------------------
            # Boosting loop:
            # -----------------------------------------
            for num_node in range(1,num_nodes):

                # Fit new boostedNode:
                boostedNode_name = "regr_L%dG%dN%d" %(self.num_layers,indx_subset,num_node)
                self.init_boostedNode(boostedNode_name, kwargs_boosted_nodes, greedyLayer_name)
                train_flag = self.train_boostedNode(boostedNode_name,  fit_boosted_nodes, indx_subset, num_node)

                # Insert in greedyLayer:
                if train_flag or self.check_insert_weights(greedyLayer_name, num_node):
                    self.convSoftmax[greedyLayer_name].insert_weights(self.regr[boostedNode_name])

                # Finetune active nodes in greedyLayer:
                self.finetune_greedyLayer(greedyLayer_name, indx_subset, finetune_nodes, num_node)

            nodes_subsets.append(self.convSoftmax[greedyLayer_name])

        # Add new greedy layer:
        self._insert_new_layer(nodes_subsets[0])
        self.pickle_greedyNET()


    def check_insert_weights(self, greedyLayer_name, active_nodes):
        '''
        It checks if to insert the weights of a boosted node into the greedyLayer.
        '''
        insert_weights = True
        if greedyLayer_name in self.preLoad:
            load = self.preLoad[greedyLayer_name]
            insert_weights = load[1]
            if len(load)==3:
                insert_weights = True if active_nodes>=load[2] else False #MOD
        return insert_weights

    def init_boostedNode(self, net_name, kwargs, convSoftmax_name):
        # If not preLoaded, initialize new node:
        if net_name not in self.regr:
            params = deepcopy(kwargs)
            logs_path = self.BASE_PATH_LOG_MODEL+net_name+'/'
            utils.create_dir(logs_path)
            params['log_frequency'] = 1
            params['subLog_filename'] = 'sub_log.txt'
            params['log_filename'] = 'log.txt'
            # params['pickleModel_mode'] = 'on_epoch_finished'
            params['pickle_filename'] = 'model.pickle'
            params['trackWeights_pdfName'] = 'weights.txt'
            params['logs_path'] = logs_path

            self.regr[net_name] = self.boosted_node_class(
                self.convSoftmax[convSoftmax_name],
                **params
            )

    def init_greedyLayer(self, net_name, kwargs, num_nodes):
        # If not preLoaded, initialize new network:
        if net_name not in self.convSoftmax:
            logs_path = self.BASE_PATH_LOG_MODEL+net_name+'/'
            utils.create_dir(logs_path)
            params = deepcopy(kwargs)
            params['log_frequency'] = 1
            params['log_filename'] = 'log.txt'
            params['subLog_filename'] = 'sub_log.txt'
            # params['pickleModel_mode'] = 'on_epoch_finished'
            params['trackWeights_pdfName'] = 'weights.txt'
            params['pickle_filename'] = 'model.pickle'
            params['logs_path'] = logs_path
            self.convSoftmax[net_name] = self.greedy_layer_class(
                self.net,
                self.output_channels,
                num_nodes=num_nodes,
                **params
            )

    def train_boostedNode(self, net_name, fit_routine, idx_net2, num_node):
        train = True
        # Check if to train:
        if net_name in self.preLoad:
            train = self.preLoad[net_name][1]

        # Train node:
        if train:
            print "\n\n----------------------------------------------------"
            print "TRAINING Boosted Softmax - Layer %d, group %d, node %d" %(self.num_layers,idx_net2,num_node)
            print "----------------------------------------------------\n\n"
            self.regr[net_name].net.first_node = True if num_node==0 else False
            self.regr[net_name].net = fit_routine(self.regr[net_name].net)
            self.post_Training(net_name)

        return train


    def finetune_greedyLayer(self, net_name, idx_net2, fit_routine=None,active_nodes=0):
        # Check if to finetune:
        train = True
        if net_name in self.preLoad:
            load = self.preLoad[net_name]
            train = load[1]
            if len(load)==3:
                train = True if active_nodes>load[2] else False

        # Finetune:
        if train:
            print "\n\n--------------------------------------------------------"
            print "FINE-TUNING Softmax - Layer %d, group %d, active nodes %d " %(self.num_layers,idx_net2,active_nodes)
            print "--------------------------------------------------------\n\n"
            if fit_routine:
                self.convSoftmax[net_name].net = fit_routine(self.convSoftmax[net_name].net)
            self.post_Training(net_name)


    def post_Training(self, net_name):
        '''
        Function called after each training of a subNetwork
        '''
        if 'cnv' in net_name:
            utils.pickle_model(self.convSoftmax[net_name],self.BASE_PATH_LOG_MODEL+net_name+'/routine.pickle')
        elif 'reg' in net_name:
            utils.pickle_model(self.regr[net_name],self.BASE_PATH_LOG_MODEL+net_name+'/routine.pickle')
        self.pickle_greedyNET()



    def _insert_new_layer(self, net2):
        '''
        Insert new computed layer, recompile main Greedy net and insert old weights.

        All parameters are trainable by default.

        '''
        # -----------------
        # Collect weights:
        # -----------------

        prevLayers_weights = get_all_param_values(self.net.layers_[self.last_layer_name])
        net2_weights = get_all_param_values(net2.net.layers_['conv1'])
        net2_weights_newNode = get_all_param_values(net2.net.layers_['conv1_newNode'])

        # -----------------
        # Add new layer:
        # -----------------
        self.num_layers += 1
        self.layers += [
            (layers.Conv2DLayer, {
                'name': 'conv%d' %(self.num_layers),
                'num_filters': net2.num_nodes*net2.num_filters1,
                'filter_size': net2.filter_size1,
                'pad':'same',
                'nonlinearity': rectify}),
        ]

        # ------------------
        # Recompile network:
        # ------------------
        self.net = segmNeuralNet(
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
        nNodes = net2.num_filters1
        net2_weights[0][nNodes*(net2.num_nodes-1):,:,:] = net2_weights_newNode[0]
        set_all_param_values(self.net.layers_[self.last_layer_name], prevLayers_weights+net2_weights)

        self.output_channels = self.net.layers[-1][1]['num_filters']


    '''
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





