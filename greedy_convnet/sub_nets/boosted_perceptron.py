__author__ = "abailoni"

'''
Network that will train one perceptron
'''
import numpy as np
from copy import deepcopy
# import json

import theano.tensor as T
from lasagne import layers
from lasagne.nonlinearities import rectify
from lasagne.init import Normal


import mod_nolearn.segm.segm_utils as segm_utils
from mod_nolearn.segm import segmNeuralNet
from greedy_convnet import BatchIterator_Greedy
import various.utils as utils

class boostedPerceptron(object):
    '''
    # ---------------------------------------------
    # STRUCTURE OF THE CLASS: (main attributes)
    # ---------------------------------------------

        - net:
            TYPE: istance of NeuralNet() in mod_nolearn
            VALUE: attribute containing the actual perceptron network that will be trained

        It inherits most of the attributes from the greedyLayer passed during
        initializition.

    '''

    def __init__(self,
            fixed_input_layers,
            layers_info,
            name_trained_layer,
            filters_in_perceptron,
            greedyLayer,
            fixed_weights,
            **greedy_kwargs):

        '''
        # ----------
        # INPUTS:
        # ----------

            - fixed_input_layers:
                TYPE: dictionary
                VALUE: nolearn-dictionary with previous layers that won't be trained

            - fixed_weights:
                TYPE: dictionary
                VALUES: weights for the fixed layers (check attribute trained_weights in class greedyNet)

            - layers_info:
                TYPE: dictionary
                VALUE: infos about all layers (e.g. output_shape, req, trained_greedily, trained, type...)

            - name_trained_layer
                TYPE: string
                VALUE: name of the trained layer (mainly for log and to check infos)

            - filters_in_perceptron:
                TYPE: int
                VALUES: number of filters in perceptron

            - greedyLayer:
                TYPE: instance of greedyLayer()
                VALUE: instance representing the full greedy trained layer
        '''


        # info = deepcopy(greedy_kwargs)

        # --------------------------
        # Inherited by greedyLayer:
        # --------------------------
        self.filter_size = greedyLayer.filter_size
        self.num_classes = greedyLayer.num_classes
        self.eval_size = greedyLayer.eval_size
        self.init_weight = greedyLayer.init_weight
        self.best_classifier = greedyLayer.net if greedyLayer.active_perceptrons!=0 else None
        self.layer_kwargs = deepcopy(greedyLayer.layer_kwargs)
        self.layer_kwargs['num_filters'] = filters_in_perceptron
        self.layer_output_shape = greedyLayer.layer_output_shape
        self.batch_size = greedyLayer.batch_size
        self.batchShuffle = greedyLayer.batchShuffle
        self.FULLNET_kwargs = greedyLayer.FULLNET_kwargs
        if self.FULLNET_kwargs:
            greedy_kwargs = utils.join_dict(self.FULLNET_kwargs, greedy_kwargs)

        # -----------------
        # Own attributes:
        # -----------------
        self.filters_in_perceptron = filters_in_perceptron
        greedy_kwargs['name'] = "greedy_"+name_trained_layer+"_perceptron"

        # -----------------
        # Input processing:
        # -----------------
        out_spatial_shape = list(layers_info[name_trained_layer]['output_shape'])[-2:]

        out_shape = [self.batch_size]+out_spatial_shape
        objective_loss_function = categorical_crossentropy_segm_boost(out_shape)

        customBatchIterator = BatchIterator_boostRegr(
            self.best_classifier,
            objective_loss_function,
            layer_output_shape=self.layer_output_shape,
            batch_size=self.batch_size,
            shuffle=self.batchShuffle,
        )


        # -----------------
        # Building NET:
        # -----------------
        self.layer_kwargs['name'] = 'greedyConv_1'
        self.layer_type = layers_info[name_trained_layer]['type']
        netLayers = deepcopy(fixed_input_layers)
        if self.layer_type=="conv":
            netLayers += [
                (layers.Conv2DLayer, self.layer_kwargs),
                (layers.Conv2DLayer, {
                    'name': 'greedyConv_2',
                    'num_filters': self.num_classes,
                    'filter_size': self.filter_size,
                    'pad':'same',
                    'W': Normal(std=self.init_weight),
                    'nonlinearity': segm_utils.softmax_segm})]

        elif self.layer_type=="trans_conv":
            netLayers += [
                (layers.TransposedConv2DLayer, self.layer_kwargs),
                (layers.Conv2DLayer, {
                    'name': 'greedyConv_2',
                    'num_filters': self.num_classes,
                    'filter_size': self.filter_size,
                    # 'crop': self.layer_kwargs['crop'],
                    'pad': 'same',
                    'W': Normal(std=self.init_weight),
                    'nonlinearity': segm_utils.softmax_segm})]

        self.net = segmNeuralNet(
            layers=netLayers,
            batch_iterator_train = customBatchIterator,
            batch_iterator_test = customBatchIterator,
            objective_loss_function = objective_loss_function,
            scores_train = [('trn pixelAcc', segm_utils.pixel_accuracy)],
            y_tensor_type = T.ltensor3,
            eval_size=self.eval_size,
            regression = False,
            **greedy_kwargs
        )

        self.net._output_layer = self.net.initialize_layers()

        # Set all fixed layers as not trainable & regularizable:
        fixed_layers_names = [layer[1]['name'] for layer in fixed_input_layers]
        for name in fixed_layers_names:
            if layers_info[name]['type']=='conv' or layers_info[name]['type']=='trans_conv':
                self.net.layers_[name].params[self.net.layers_[name].W].remove('trainable')
                self.net.layers_[name].params[self.net.layers_[name].b].remove('trainable')
                self.net.layers_[name].params[self.net.layers_[name].W].remove('regularizable')

        # Set last greedy-layer as not regularizable:
        self.net.layers_['greedyConv_2'].params[self.net.layers_['greedyConv_2'].W].remove('regularizable')


        self.net.initialize()


        self.insert_weights_fixedLayers(fixed_weights)


        # # -------------------------------------
        # # SAVE INFO NET: (for log)
        # # -------------------------------------
        # info['filter_size1'] = self.filter_size1
        # info['filter_size2'] = self.filter_size2
        # info['input_filters'] = self.input_filters
        # info['num_filters1'] = self.num_filters1
        # info['num_classes'] = self.num_classes
        # info['xy_input'] = self.xy_input
        # info['eval_size'] = self.eval_size

        # info.pop('update', None)
        # info.pop('on_epoch_finished', None)
        # info.pop('on_batch_finished', None)
        # info.pop('on_training_finished', None)
        # info.pop('noReg_loss', None)
        # for key in [key for key in info if 'update_' in key]:
        #     info[key] = info[key].get_value().item()
        # json.dump(info, file(info['logs_path']+'/info-net.txt', 'w'))


    def insert_weights_fixedLayers(self, fixed_weights):
        '''
        The function inserts the weights

        fixed_weights
            type: dictionary
            value: key->layer_name; content -> list with paramsValues
        '''
        for layer in self.net.layers_:
            if layer not in ["greedyConv_1", "greedyConv_2", "mask"]:
                params_fixed_layer = self.net.layers_[layer].get_params()
                if layer in fixed_weights:
                    for i, param in enumerate(params_fixed_layer):
                        param.set_value(fixed_weights[layer][i])
                elif len(params_fixed_layer)!=0:
                    raise ValueError("Weights of fixed layer %s not available!" %(layer))




class BatchIterator_boostRegr(BatchIterator_Greedy):
    '''
    Given a batch of images and the targets, it updates the boosting weights given
    by the residuals of the previous classifier.
    '''
    def __init__(self, best_classifier, objective_boost_loss, *args, **kwargs):
        self.best_classifier = best_classifier
        self.objective_boost_loss = objective_boost_loss
        super(BatchIterator_boostRegr, self).__init__(*args, **kwargs)
        # Avoid to ignore completely the pixel that were rightly predicted:
        self.residual_constant = 0.8

    def transform(self, Xb, yb):
        # First of all rezise the ground truth to the right scale:
        Xb, yb = super(BatchIterator_boostRegr, self).transform(Xb, yb)


        if yb is not None:
            if self.best_classifier:
                # Updates boosting weights:
                pred_proba = self.best_classifier.predict_proba(Xb) #[N,C,x,y]
                C = pred_proba.shape[1]
                pred_proba = pred_proba.transpose((0,2,3,1)).reshape((-1,C)).astype(np.float32)
                yb_vect = yb.reshape((-1,))
                prob_residuals = 1. - pred_proba[np.arange(yb_vect.shape[0]),yb_vect] * self.residual_constant
                self.objective_boost_loss.update_boosting_weights(prob_residuals)
            else:
                shape = self.objective_boost_loss.shape
                self.objective_boost_loss.update_boosting_weights(np.ones((shape,),dtype=np.float32))
        return Xb, yb

from theano.tensor.nnet import categorical_crossentropy
import theano

class categorical_crossentropy_segm_boost(object):
    '''
    Modified categorical_crossentropy loss for the boosting weights.
    '''
    def __init__(self, out_shape):
        self.shape = out_shape[0]*out_shape[1]*out_shape[2]
        self.boosting_weights = theano.shared(np.ones((self.shape,),dtype=np.float32))

    def update_boosting_weights(self,prob_residuals):
        self.boosting_weights.set_value(prob_residuals)

    def __call__(self, prediction_proba, targets):
        if not self.boosting_weights:
            raise ValueError("Boosting residuals not set up")

        shape = T.shape(prediction_proba)
        pred_mod1 = T.transpose(prediction_proba, (0,2,3,1))
        pred_mod = T.reshape(pred_mod1, (-1,shape[1]))
        if prediction_proba.ndim == targets.ndim:
            targ_mod1 = T.transpose(targets,(0,2,3,1))
            targ_mod = T.reshape(targ_mod1,(-1,shape[1]))
        else:
            targ_mod = T.reshape(targets, (-1,))
        results = categorical_crossentropy(pred_mod, targ_mod)
        results *= self.boosting_weights[:results.shape[0]]
        results = T.reshape(results, (shape[0],shape[2],shape[3]))
        return T.sum(results, axis=(1,2))



