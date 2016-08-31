import time
from copy import copy, deepcopy

import theano.tensor as T
import lasagne.layers as layers
from lasagne.layers import set_all_param_values, get_all_param_values
from lasagne.nonlinearities import rectify


import pretr_nets.vgg16 as vgg16

import mod_nolearn.nets.segmNet as segmNet
import mod_nolearn.segmentFcts as segmentFcts
import greedyNET.nets.boostRegr as boostRegr
import greedyNET.nets.convSoftmax as convSoftmax
# from greedyNET.greedy_utils import clean_kwargs
import mod_nolearn.utils as utils

from nolearn.lasagne.visualize import draw_to_file


def restore_greedyModel(model_name, path_logs='./logs/'):
    return utils.restore_model(path_logs+model_name+'/model.pickle')

class greedyRoutine(object):
    def __init__(self, num_VGG16_layers, **kwargs):
        '''
        Initialize a network that uses just the first layers of VGG16.

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
        self.model_name = kwargs.pop('model_name', 'greedyNET')
        self.BASE_PATH_LOG = kwargs.pop('BASE_PATH_LOG', "./logs/")
        self.layers = vgg16.nolearn_vgg16_layers()[:self.num_VGG16_layers+1]
        fixed_kwargs = {
            'objective_loss_function': segmNet.binary_crossentropy_segm,
            'y_tensor_type': T.ftensor3,
            'eval_size': self.eval_size,
            'regression': True
        }
        self.net_kwargs = kwargs.copy()
        self.net_kwargs.update(fixed_kwargs)

        self.net = segmNet.segmNeuralNet(
            layers=self.layers,
            scores_train=[('trn pixelAcc', segmentFcts.pixel_accuracy_sigmoid)],
            scores_valid=[('val pixelAcc', segmentFcts.pixel_accuracy_sigmoid)],
            **self.net_kwargs
        )

        # print "\n\n---------------------------"
        # print "Compiling inputProcess...\n---------------------------"
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
        self.BASE_PATH_LOG_MODEL = self.BASE_PATH_LOG+self.model_name+'/'
        self.preLoad = {}

    def update_name(self, newname):
        self.model_name = newname
        self.BASE_PATH_LOG_MODEL = self.BASE_PATH_LOG+self.model_name+'/'

    def train_new_layer(self, (fit_routine_regr, num_regr, kwargs_regr) , (fit_routine_net2, num_net2, kwargs_convSoftmax), finetune_routine_net2):

        # ------------------------------------------------
        # Parallelized loop: (not working for the moment)
        # ------------------------------------------------
        self.best_classifier = {}
        Nets2 = []
        for idx_net2 in range(num_net2):
            # -------------------------------------------
            # Fit first node and initialize convSoftmax:
            # -------------------------------------------
            group_label = "group_%d" %idx_net2
            convSoftmax_name = "cnv_L%d_G%d"%(self.num_layers,idx_net2)
            self.init_convSoftmax(convSoftmax_name, kwargs_convSoftmax, num_regr)
            self.train_convSoftmax(convSoftmax_name, fit_routine_net2, finetune_routine_net2, idx_net2, 0)
            self.best_classifier[group_label] = self.convSoftmax[convSoftmax_name].net

            # -----------------------------------------
            # Boosting loop:
            # -----------------------------------------
            for num_node in range(1,num_regr):
                # Fit new regression to residuals:
                regr_name = "regr_L%dG%dN%d" %(self.num_layers,idx_net2,num_node)
                self.init_regr(regr_name, kwargs_regr, convSoftmax_name)
                self.train_regr(regr_name, fit_routine_regr, idx_net2, num_node)


                # Insert in convSoftmax and re-train:
                self.convSoftmax[convSoftmax_name].insert_weights(self.regr[regr_name])
                self.train_convSoftmax(convSoftmax_name, fit_routine_net2, finetune_routine_net2, idx_net2, num_node)

            Nets2.append(self.convSoftmax[convSoftmax_name]) ## This should be ok

        # Add new layer:
        self._insert_new_layer(Nets2[0])
        self.pickle_greedyNET()

    def load_nodes(self, preLoad):
        '''
        Loads the specifications in order to load specific pretrained pickled nodes.
        The input is a dictionary such that:
            - the keys are the names of the nodes (e.g. convL0G0)
            - each element contains a tuple such that:
                 (name_model_from_which_import, train_flag) --> (None, True)
        '''
        self.preLoad = preLoad


    def pickle_greedyNET(self):
        utils.pickle_model(self, self.BASE_PATH_LOG_MODEL+'model.pickle')

    def init_regr(self, net_name, kwargs, convSoftmax_name):
        if net_name in self.preLoad:
            load = self.preLoad[net_name]
            if load[0]:
                # Imported from other pretrained model:
                model_path = self.BASE_PATH_LOG+load[0]+'/'+net_name+'/model.pickle'
                self.regr[net_name] = utils.restore_model(model_path)
                # Copy folder and logs in new model:
                utils.create_dir(self.BASE_PATH_LOG_MODEL+net_name)
                utils.copyDirectory(self.BASE_PATH_LOG+load[0]+'/'+net_name, self.BASE_PATH_LOG_MODEL+net_name)
        elif net_name not in self.regr:
            # Initialize new network:
            params = deepcopy(kwargs)
            logs_path = self.BASE_PATH_LOG_MODEL+net_name+'/'
            utils.create_dir(logs_path)
            params['subLog_filename'] = logs_path+'sub_log.txt'
            params['log_filename'] = logs_path+'log.txt'
            params['pickleModel_mode'] = 'on_epoch_finished'
            params['pickle_filename'] = logs_path+'model.pickle'
            params['trackWeights_pdfName'] = logs_path+'weights.txt'
            params['logs_path'] = logs_path

            self.regr[net_name] = boostRegr.boostRegr_routine(
                self.convSoftmax[convSoftmax_name],
                **params
            )

    def init_convSoftmax(self, net_name, kwargs, num_nodes):
        if net_name in self.preLoad:
            load = self.preLoad[net_name]
            if load[0]:
                # Imported from other pretrained model:
                model_path = self.BASE_PATH_LOG+load[0]+'/'+net_name+'/model.pickle'
                self.convSoftmax[net_name] = utils.restore_model(model_path)
                # Copy folder and logs in new model:
                utils.create_dir(self.BASE_PATH_LOG_MODEL+net_name)
                utils.copyDirectory(self.BASE_PATH_LOG+load[0]+'/'+net_name, self.BASE_PATH_LOG_MODEL+net_name)
        elif net_name not in self.convSoftmax:
            logs_path = self.BASE_PATH_LOG_MODEL+net_name+'/'

            logs_path = logs_path+net_name+'/'
            utils.create_dir(logs_path)
            params = deepcopy(kwargs)
            params['log_path'] = logs_path+net_name+'/'
            params['log_filename'] = logs_path+'log.txt'
            params['subLog_filename'] = logs_path+'sub_log.txt'
            params['pickleModel_mode'] = 'on_training_finished'
            params['trackWeights_pdfName'] = logs_path+'weights.txt'
            params['pickle_filename'] = logs_path+'model.pickle'
            params['logs_path'] = logs_path
            self.convSoftmax[net_name] = convSoftmax.convSoftmax_routine(
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
            print "TRAINING regression - Layer %d, group %d, node %d" %(self.num_layers,idx_net2,num_node)
            print "----------------------------------------------------\n\n"
            self.regr[net_name].net = fit_routine(self.regr[net_name].net)
            self.post_Training(net_name)


    def train_convSoftmax(self, net_name, fit_routine, finetune_routine, idx_net2, active_nodes=0):
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
            print "TRAINING softmax - Layer %d, group %d, %d active nodes" %(self.num_layers,idx_net2,active_nodes)
            print "--------------------------------------------------------\n\n"
            print "Tuning new node:"
            self.convSoftmax[net_name].net = fit_routine(self.convSoftmax[net_name].net)
            if active_nodes!=0:
                self.convSoftmax[net_name].activate_nodes()
                print "\nFinetuning all second layer until last node:"
                self.convSoftmax[net_name].net = finetune_routine(self.convSoftmax[net_name].net)
            self.post_Training(net_name)


    def post_Training(self, net_name):
        '''
        Function called after each training of a subNetwork
        '''
        self.pickle_greedyNET()

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
        net2_weights = get_all_param_values(net2.net.layers_['conv1'])
        net2_weights_newNode = get_all_param_values(net2.net.layers_['conv1_newNode'])

        # -----------------
        # Add new layer:
        # -----------------
        self.num_layers += 1
        self.layers = self.layers + [
            (layers.Conv2DLayer, {
                'name': 'conv%d' %(self.num_layers),
                'num_filters': net2.num_classes*net2.num_nodes*net2.num_filters_regr,
                'filter_size': net2.filter_size1,
                'pad':'same',
                'nonlinearity': rectify}),
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
        nNodes = net2.num_classes*net2.num_filters_regr
        net2_weights[0][nNodes*(net2.num_nodes-1):,:,:] = net2_weights_newNode[0]
        set_all_param_values(self.net.layers_[self.last_layer_name], prevLayers_weights+net2_weights)

        self.output_channels = self.net.layers[-1][1]['num_filters']




