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


class boostedPerceptron(object):
    def __init__(self,
            fixed_input_layers,
            layers_info,
            name_trained_layer,
            trained_layer_args,
            filters_in_perceptron,
            greedyLayer,
            **kwargs):
        # info = deepcopy(kwargs)
        # --------------------------
        # Inherited by convSoftmax:
        # --------------------------
        self.filter_size1 = greedyLayer.filter_size1
        self.filter_size2 = greedyLayer.filter_size2
        self.num_filters1 = greedyLayer.num_filters1
        self.num_classes = greedyLayer.num_classes
        self.eval_size = greedyLayer.eval_size
        self.init_weight = greedyLayer.init_weight
        self.best_classifier = greedyLayer.net if greedyLayer.active_perceptrons!=0 else None
        self.GT_output_shape = greedyLayer.GT_output_shape

        # -----------------
        # Own attributes:
        # -----------------
        self.filters_in_perceptron = filters_in_perceptron
        kwargs['name'] = "greedy_"+name_trained_layer+"_perceptron"
        self.batch_size = kwargs.pop('batch_size', 20)
        self.batchShuffle = kwargs.pop('batchShuffle', True)

        # -----------------
        # Input processing:
        # -----------------
        out_shape = list(layers_info[name_trained_layer]['output_shape'])

        print out_shape
        out_shape[0] = 1000
        print "ATTENTION: first dimension is fake..."

        objective_loss_function = categorical_crossentropy_segm_boost(out_shape)

        customBatchIterator = BatchIterator_boostRegr(
            self.best_classifier,
            objective_loss_function,
            GT_output_shape=self.GT_output_shape,
            batch_size=self.batch_size,
            shuffle=self.batchShuffle,
        )

        # -----------------
        # Building NET:
        # -----------------
        self.layer_type = layers_info[name_trained_layer]['type']
        netLayers = deepcopy(fixed_input_layers)
        if self.layer_type=="conv":
            netLayers += [
                (layers.Conv2DLayer, {
                    'name': "greedyConv_1",
                    'num_filters': filters_in_perceptron,
                    'filter_size': self.filter_size1,
                    'pad':'same',
                    # 'W': Normal(std=1000),
                    'nonlinearity': rectify}),
                (layers.Conv2DLayer, {
                    'name': 'greedyConv_2',
                    'num_filters': self.num_classes,
                    'filter_size': self.filter_size2,
                    'pad':'same',
                    # 'W': Normal(std=1000),
                    'nonlinearity': segm_utils.softmax_segm})]

        elif self.layer_type=="trans_conv":
            netLayers += [
                (layers.TrasposedConv2DLayer, {
                    'name': "greedyConv_1",
                    'num_filters': filters_in_perceptron,
                    'filter_size': self.filter_size1,
                    'crop': trained_layer_args['crop'],  ## check if crop is always there
                    # 'W': Normal(std=1000),
                    'nonlinearity': rectify}),
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
            objective_loss_function = objective_loss_function,
            scores_train = [('trn pixelAcc', segm_utils.pixel_accuracy)],
            y_tensor_type = T.ltensor3,
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

        # Set last greedy-layer as not regularizable:
        self.net.layers_['greedyConv_2'].params[self.net.layers_['greedyConv_2'].W].remove('regularizable')


        self.net.initialize()

        # # -------------------------------------
        # # SAVE INFO NET:
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
        Xb, yb = super(BatchIterator_boostRegr, self).transform(Xb, yb)
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
        # if not self.boosting_weights:
        #     self.boosting_weights = theano.shared(prob_residuals)
        # else:
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



