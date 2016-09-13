import time
from copy import deepcopy

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
    # After this, if you rename the model, you should call the method update_all_paths.
    return utils.restore_model(path_logs+model_name+'/model.pickle')

class greedyNet(object):
    def __init__(self, num_VGG16_layers, mod=None, **kwargs):
        '''
        Initialize a network that uses just the first layers of VGG16.

        Inputs:
         - How many layers to keep of VGG16 (int).
         - eval_size (0.1)

        Be careful with Pooling.

        All the arguments are the usual for nolearn NeuralNet (segmNet)
        '''
        # ------------------------------------
        # Compile network with vgg16 layers:
        # ------------------------------------
        self.num_layers = 0
        if mod=='basic':
            self.greedy_layer_class = greedyLayer
            self.boosted_node_class = boostedNode
        elif mod=='ReLU':
            self.greedy_layer_class = greedyLayer_ReLU
            self.boosted_node_class = boostedNode_ReLU
        else:
            raise ValueError("A mode is required. Accepted types: 'basic', 'ReLU'")

        self.num_VGG16_layers = int(num_VGG16_layers)
        self.eval_size =  kwargs.pop('eval_size',0.)
        self.model_name = kwargs.pop('model_name', 'greedyNET')
        self.BASE_PATH_LOG = kwargs.pop('BASE_PATH_LOG', "./logs/")
        self.BASE_PATH_LOG_MODEL = self.BASE_PATH_LOG+self.model_name+'/'
        self.layers = vgg16.nolearn_vgg16_layers()[:self.num_VGG16_layers+1]
        fixed_kwargs = {
            'objective_loss_function': segm_utils.categorical_crossentropy_segm,
            'y_tensor_type': T.ltensor3,
            'eval_size': self.eval_size,
            'regression': False,
            'logs_path': self.BASE_PATH_LOG_MODEL
        }
        self.net_kwargs = kwargs.copy()
        self.net_kwargs.update(fixed_kwargs)

        self.net = segmNeuralNet(
            layers=self.layers,
            scores_train=[('trn pixelAcc', segm_utils.pixel_accuracy)],
            scores_valid=[('val pixelAcc', segm_utils.pixel_accuracy)],
            **self.net_kwargs
        )

        # print "Compiling inputProcess..."
        # tick = time.time()
        self.net.initialize()
        # tock = time.time()
        # print "Done! (%f sec.)\n\n\n" %(tock-tick)


        # --------------------
        # Copy vgg16 weights:
        # --------------------
        self.net = vgg16.nolearn_insert_weights_vgg16(self.net,self.num_VGG16_layers)

        self.output_channels = self.net.layers[-1][1]['num_filters']
        self.last_layer_name = self.net.layers[-1][1]['name']

        self.regr, self.convSoftmax = {}, {}
        self.preLoad = {}


    def train_new_layer(self, (fit_routine_regr, num_regr, kwargs_regr) , (fit_routine_net2, num_net2, kwargs_convSoftmax), finetune_routine_net2, adjust_convSoftmax=None):

        # ------------------------------------------------
        # Parallelized loop: (serialized for the moment)
        # ------------------------------------------------
        Nets2 = []
        for idx_net2 in range(num_net2):
            # -------------------------------------------
            # Fit first node and initialize convSoftmax:
            # -------------------------------------------
            convSoftmax_name = "cnv_L%d_G%d"%(self.num_layers,idx_net2)
            self.init_convSoftmax(convSoftmax_name, kwargs_convSoftmax, num_regr)

            regr_name = "regr_L%dG%dN%d" %(self.num_layers,idx_net2,0)
            self.init_regr(regr_name, kwargs_regr, convSoftmax_name)
            train_flag = self.train_regr(regr_name,  fit_routine_regr, idx_net2, 0)
            # train_flag = True


            # Check if first node was trained or it was loaded
            # but not inserted in the main net:
            if train_flag or self.convSoftmax[convSoftmax_name].net.layers_['mask'].first_iteration:
                self.convSoftmax[convSoftmax_name].insert_weights(self.regr[regr_name])
            # self.train_convSoftmax(convSoftmax_name, idx_net2, fit_routine_net2, finetune_routine_net2,  0)

            # -----------------------------------------
            # Boosting loop:
            # -----------------------------------------
            for num_node in range(1,num_regr):
                # Fit new regression to residuals:
                regr_name = "regr_L%dG%dN%d" %(self.num_layers,idx_net2,num_node)
                self.init_regr(regr_name, kwargs_regr, convSoftmax_name)

                train_flag = self.train_regr(regr_name,  fit_routine_regr, idx_net2, num_node)
                # train_flag=True

                # Insert in convSoftmax:
                if train_flag or self.check_insert_weights(convSoftmax_name, num_node):
                    self.convSoftmax[convSoftmax_name].insert_weights(self.regr[regr_name])

                # # Call external function to adjust parameters:
                # if adjust_convSoftmax:
                #     self.convSoftmax[convSoftmax_name] = adjust_convSoftmax(self.convSoftmax[convSoftmax_name])

                # Retrain:
                self.train_convSoftmax(convSoftmax_name, idx_net2, fit_routine_net2, finetune_routine_net2, num_node)

            Nets2.append(self.convSoftmax[convSoftmax_name]) ## This should be ok

        # Add new layer:
        self._insert_new_layer(Nets2[0])
        self.pickle_greedyNET()


    def check_insert_weights(self, net_name, active_nodes):
        # Really bad repetition of code... BUT THERE IS MOD IN SIGN!
        insert_weights = True
        if net_name in self.preLoad:
            load = self.preLoad[net_name]
            insert_weights = load[1]
            if len(load)==3:
                print active_nodes, load[2]
                insert_weights = True if active_nodes>=load[2] else False #MOD
        return insert_weights

    def init_regr(self, net_name, kwargs, convSoftmax_name):
        # If not pretrained, initialize new network:
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

    def init_convSoftmax(self, net_name, kwargs, num_nodes):
        # If not pretrained, initialize new network:
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

    def train_regr(self, net_name, fit_routine, idx_net2, num_node):
        train = True
        # Check if to train:
        if net_name in self.preLoad:
            train = self.preLoad[net_name][1]

        # Train subNet:
        if train:
            print "\n\n----------------------------------------------------"
            print "TRAINING Boosted Softmax - Layer %d, group %d, node %d" %(self.num_layers,idx_net2,num_node)
            print "----------------------------------------------------\n\n"
            self.regr[net_name].net.first_node = True if num_node==0 else False
            self.regr[net_name].net = fit_routine(self.regr[net_name].net)
            self.post_Training(net_name)

        return train


    def train_convSoftmax(self, net_name, idx_net2, fit_routine=None, finetune_routine=None,  active_nodes=0):
        # Check if to train:
        train = True
        if net_name in self.preLoad:
            load = self.preLoad[net_name]
            train = load[1]
            if len(load)==3:
                train = True if active_nodes>load[2] else False

        # Train subNet:
        if train:
            print "\n\n--------------------------------------------------------"
            print "FINE-TUNING Softmax - Layer %d, group %d, active nodes %d " %(self.num_layers,idx_net2,active_nodes)
            print "--------------------------------------------------------\n\n"
            # if active_nodes!=0:
            #     self.convSoftmax[net_name].deactivate_nodes()
            if fit_routine:
                # print "Tuning new node:"
                self.convSoftmax[net_name].net = fit_routine(self.convSoftmax[net_name].net)
            # if active_nodes!=0:
            #     self.convSoftmax[net_name].activate_nodes()
            # if finetune_routine:
            #     print "\nFinetuning all second layer until last node:"
            #     self.convSoftmax[net_name].net = finetune_routine(self.convSoftmax[net_name].net)
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
        Insert new computed layer, recompile and insert old weights.

        All parameters are trainable by default.

        IT SHOULD BE UPDATED TAKING A LIST OF NET2 COMPUTED IN PARALLEL...
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


    # -------------------
    # UTILS METHODS:
    # -------------------
    # To import previously pretrained full or partial Greedy models:
    #
    #   - create a new GreedyNET or import existing one using
    #     restore_greedyModel()
    #
    #   - in order not to overwrite the old model, call update_all_paths()
    #     to update the name (and change the logs path)
    #
    #   - if subNets should be loaded, call load_subNets()


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
                - the keys are the names of the nodes (e.g. convL0G0)
                - each element contains a tuple such that:
                     (path_to_pretr_model, train_flag)
                     ('logs/model_A/', True, nodes_trained)

        The third options indicates the pretrained nodes of a convSoftmax subNet.
        In particular we have:
                train = True if active_nodes>load[2] else False
        where the active_nodes are the one in the MAIN part of the net, w/o considering the new node.
        Conclusion: just put how many nodes have been aleardy trained in the MAIN part of the net (given by convSoft.active_nodes)

        What it does in order:
            - import subNets
            - copy folders/logs in main model (and delete previous versions)
            - update all paths of subNets
        '''
        import mod_nolearn.utils as utils
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



