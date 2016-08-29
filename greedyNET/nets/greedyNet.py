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

    def train_new_layer(self, (fit_routine_LogRegr, num_regr, kwargs_logRegr) , (fit_routine_net2, num_net2, kwargs_net2), finetune_routine_net2):

        # ------------------------------------------------
        # Parallelized loop: (not working for the moment)
        # ------------------------------------------------
        Nets2 = []
        for idx_net2 in range(num_net2):
            # -----------------------------------------
            # Fit first LogRegr:
            # -----------------------------------------
            lgRgrNet_name = "rgr_L%dG%dN0" %(self.num_layers,idx_net2)
            init, train = self.before_Training(lgRgrNet_name)
            if init:
                params_logRegr = deepcopy(kwargs_logRegr)
                logs_path = self.BASE_PATH_LOG_MODEL+lgRgrNet_name+'/'
                utils.create_dir(logs_path)
                params_logRegr['subLog_filename'] = logs_path+'sub_log.txt'
                params_logRegr['log_filename'] = logs_path+'log.txt'
                params_logRegr['pickleModel_mode'] = 'on_epoch_finished'
                params_logRegr['pickle_filename'] = logs_path+'model.pickle'
                params_logRegr['trackWeights_pdfName'] = logs_path+'weights.txt'

                self.regr[lgRgrNet_name] = boostRegr.boostRegr_routine(
                    self.net,
                    self.output_channels,
                    best_classifier=None,
                    **params_logRegr
                )
            if train:
                print "\n\n----------------------------------------------------"
                print "TRAINING regression - Layer %d, group %d, node 0" %(self.num_layers,idx_net2)
                print "----------------------------------------------------\n\n"
                self.regr[lgRgrNet_name].net = fit_routine_LogRegr(self.regr[lgRgrNet_name].net)
                self.post_Training(lgRgrNet_name)
            self.best_classifier = self.regr[lgRgrNet_name].net

            # -----------------------------------------
            # Initialize Net2:
            # -----------------------------------------
            convNet_name = "cnv_L%d_G%d"%(self.num_layers,idx_net2)
            init, _ = self.before_Training(convNet_name)
            if init:
                logs_path = self.BASE_PATH_LOG_MODEL+convNet_name+'/'
                utils.create_dir(logs_path)
                params_net2 = deepcopy(kwargs_net2)
                params_net2['log_filename'] = logs_path+'log.txt'
                params_net2['subLog_filename'] = logs_path+'sub_log.txt'
                params_net2['pickleModel_mode'] = 'on_training_finished'
                params_net2['trackWeights_pdfName'] = logs_path+'weights.txt'
                params_net2['pickle_filename'] = logs_path+'model.pickle'
                self.convSoftmax[convNet_name] = convSoftmax.convSoftmax_routine(
                    self.regr[lgRgrNet_name],
                    num_nodes=num_regr,
                    **params_net2
                )

                draw_to_file(self.convSoftmax[convNet_name].net,"conv_mergeNet.pdf")

            # -----------------------------------------
            # Boosting loop:
            # -----------------------------------------
            for num_node in range(1,num_regr):
                # Fit new logRegr to residuals:
                lgRgrNet_name = "lgRgr_L%dG%dN%d" %(self.num_layers,idx_net2,num_node)
                init, train = self.before_Training(lgRgrNet_name)
                if init:
                    params_logRegr = deepcopy(kwargs_logRegr)
                    logs_path = self.BASE_PATH_LOG_MODEL+lgRgrNet_name+'/'
                    utils.create_dir(logs_path)
                    params_logRegr['log_filename'] = logs_path+'log.txt'
                    params_logRegr['subLog_filename'] = logs_path+'sub_log.txt'
                    params_logRegr['pickleModel_mode'] = 'on_epoch_finished'
                    params_logRegr['trackWeights_pdfName'] = logs_path+'weights.txt'
                    params_logRegr['pickle_filename'] = logs_path+'model.pickle'
                    self.regr[lgRgrNet_name] = boostRegr.boostRegr_routine(
                        self.net,
                        self.output_channels,
                        best_classifier=self.best_classifier,
                        **params_logRegr
                    )
                if train:
                    print "\n\n----------------------------------------------------"
                    print "TRAINING regression - Layer %d, group %d, node %d" %(self.num_layers,idx_net2,num_node)
                    print "----------------------------------------------------\n\n"
                    self.regr[lgRgrNet_name].net = fit_routine_LogRegr(self.regr[lgRgrNet_name].net)
                    self.post_Training(lgRgrNet_name)
                    self.convSoftmax[convNet_name].insert_weights(self.regr[lgRgrNet_name])

                # Fit convNet:
                _, train_convNet = self.before_Training(convNet_name, node=num_node+1)
                if train or train_convNet:
                    print "\n\n--------------------------------------------------------"
                    print "TRAINING Conv. layer - Layer %d, group %d, %d active nodes" %(self.num_layers,idx_net2,num_node+1)
                    print "--------------------------------------------------------\n\n"
                    print "Tuning new node:"
                    self.convSoftmax[convNet_name].net = fit_routine_net2(self.convSoftmax[convNet_name].net)
                    self.convSoftmax[convNet_name].activate_nodes()
                    print "\nFinetuning all second layer until last node:"
                    self.convSoftmax[convNet_name].net = finetune_routine_net2(self.convSoftmax[convNet_name].net)
                    self.post_Training(convNet_name)

                # Update the best classifier:
                self.best_classifier = self.convSoftmax[convNet_name].net

            Nets2.append(self.convSoftmax[convNet_name]) ## This should be ok

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

    def before_Training(self, net_name, **kwargs):
        '''
        Function called before each training of a subNetwork.
        It checks if the subNet can be loaded from a pickle file and if it should be trained.
        '''
        node = kwargs.pop('node', None)
        # Check if net already in greedyNet:
        init, train = True, True
        if net_name in self.regr or net_name in self.convSoftmax:
            init, train = False, False

        if node:
            train = True

        if net_name in self.preLoad:
            load = self.preLoad[net_name]
            train = load[1]
            # Check if the net should be imported from another pretrained model:
            if load[0]:
                model_path = self.BASE_PATH_LOG+load[0]+'/'+net_name+'/model.pickle'
                if 'conv' in net_name:
                    self.convSoftmax[net_name] = utils.restore_model(model_path)
                else:
                    self.regr[net_name] = utils.restore_model(model_path)

                # Copy folder and logs in new model:
                utils.create_dir(self.BASE_PATH_LOG_MODEL+net_name)
                utils.copyDirectory(self.BASE_PATH_LOG+load[0]+'/'+net_name, self.BASE_PATH_LOG_MODEL+net_name)
                # Just for conv. nets trained with a specific # of act. nodes:
                if len(load)==3:
                    train = True if node>load[2] else False
        return init, train

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




