import numpy as np
from copy import deepcopy
import json

from lasagne import layers
from lasagne.init import Normal
import theano.tensor as T


from mod_nolearn.segm.segm_utils import pixel_accuracy, softmax_segm
from mod_nolearn.segm import segmNeuralNet
from greedy_convnet import BatchIterator_Greedy


# DEFAULT_imgShape = (1024,768)



class BatchIterator_boostRegr(BatchIterator_Greedy):
    '''
    Given a batch of images and the targets, it updates the boosting weights given
    by the residuals of the previous classifier.
    '''
    def __init__(self, best_classifier, objective_boost_loss, *args, **kwargs):
        self.best_classifier = best_classifier
        self.objective_boost_loss = objective_boost_loss
        super(BatchIterator_boostRegr, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):
        if yb is not None:

            if self.best_classifier:
                # Updates boosting weights:
                pred_proba = self.best_classifier.predict_proba(Xb) #[N,C,x,y]
                C = pred_proba.shape[1]
                pred_proba = pred_proba.transpose((0,2,3,1)).reshape((-1,C)).astype(np.float32)
                yb_vect = yb.reshape((-1,))
                prob_residuals = 1. - pred_proba[np.arange(yb_vect.shape[0]),yb_vect]
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




class boostedNode(object):
    def __init__(self,greedyLayer,**kwargs):
        info = deepcopy(kwargs)
        # --------------------------
        # Inherited by greedyLayer:
        # --------------------------
        self.xy_input = greedyLayer.xy_input
        self.filter_size1 = greedyLayer.filter_size1
        self.filter_size2 = greedyLayer.filter_size2
        self.input_filters = greedyLayer.input_filters
        self.num_filters1 = greedyLayer.num_filters1
        self.num_classes = greedyLayer.num_classes
        self.previous_layers = greedyLayer.previous_layers
        self.eval_size = greedyLayer.eval_size
        self.best_classifier = greedyLayer.net if not greedyLayer.net.layers_['mask'].first_iteration else None

        if 'eval_size' in kwargs:
            if kwargs['eval_size']!=self.eval_size:
                raise ValueError("A different value of eval_size was passed to the boosted node, as compared to the value of the greedyLayer")

        # -----------------
        # Own attributes:
        # -----------------
        kwargs.setdefault('name', 'boostedNode')
        self.init_weight = kwargs.pop('init_weight', 1e-3)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.batchShuffle = kwargs.pop('batchShuffle', True)

        # -----------------
        # Input processing:
        # -----------------
        if not self.xy_input[0] or not self.xy_input[1]:
            raise ValueError("Xy dimensions required for boosting weights")
        out_shape = (self.batch_size, self.xy_input[0], self.xy_input[1])
        objective_loss_function = categorical_crossentropy_segm_boost(out_shape)
        customBatchIterator = BatchIterator_boostRegr(
            objective_boost_loss=objective_loss_function,
            batch_size=self.batch_size,
            shuffle=self.batchShuffle,
            best_classifier=self.best_classifier,
            previous_layers=self.previous_layers
        )

        # -----------------
        # Building NET:
        # -----------------
        netLayers = [
            # layer dealing with the input data
            (layers.InputLayer, {'shape': (None, self.input_filters, self.xy_input[0], self.xy_input[1])}),
            (layers.Conv2DLayer, {
                'name': 'conv1',
                'num_filters': self.num_classes,
                'filter_size': self.filter_size1,
                'pad':'same',
                'W': Normal(std=self.init_weight),
                'nonlinearity': softmax_segm}),
            # (UpScaleLayer, {'imgShape':self.imgShape}),
        ]

        self.net = segmNeuralNet(
            layers=netLayers,
            batch_iterator_train = customBatchIterator,
            batch_iterator_test = customBatchIterator,
            objective_loss_function = objective_loss_function,
            scores_train = [('trn pixelAcc', pixel_accuracy)],
            y_tensor_type = T.ltensor3,
            **kwargs
        )
        # print "\n\n---------------------------\nCompiling regr network...\n---------------------------"
        # tick = time.time()
        self.net.initialize()
        # tock = time.time()
        # print "Done! (%f sec.)\n\n\n" %(tock-tick)

        # -------------------------------------
        # SAVE INFO NET:
        # -------------------------------------
        info['filter_size1'] = self.filter_size1
        info['filter_size2'] = self.filter_size2
        info['input_filters'] = self.input_filters
        info['num_filters1'] = self.num_filters1
        info['num_classes'] = self.num_classes
        info['xy_input'] = self.xy_input
        info['eval_size'] = self.eval_size

        info.pop('update', None)
        info.pop('on_epoch_finished', None)
        info.pop('on_batch_finished', None)
        info.pop('on_training_finished', None)
        info.pop('noReg_loss', None)
        for key in [key for key in info if 'update_' in key]:
            info[key] = info[key].get_value().item()
        json.dump(info, file(info['logs_path']+'/info-net.txt', 'w'))


