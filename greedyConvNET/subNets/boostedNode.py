import numpy as np
from copy import deepcopy
import json

from lasagne import layers
from lasagne.init import Normal
import theano.tensor as T


from mod_nolearn.segm.segm_utils import pixel_accuracy, softmax_segm
from mod_nolearn.segm import segmNeuralNet
from greedyConvNET import BatchIterator_Greedy

DEFAULT_imgShape = (1024,768)

# class UpScaleLayer(layers.Layer):
#     '''
#     --------------------------
#     Subclass of lasagne.layers.Layer:
#     --------------------------

#     Upscale (if necessary) the output of the logistic regression to match the
#     dimensions of the initial image.

#     The algorithm used is bilinear interpolation.
#     The layer is not Trainable at this stage.

#     UNCOMPLETE: for the moment is an identity layer.
#
#     WRONG IMPLEMENTATION init(). Check MaskLayer...
#     '''
#     def __init__(self, *args, **kwargs):
#         self.imgShape = kwargs.pop('imgShape', DEFAULT_imgShape)
#         super(UpScaleLayer, self).__init__(*args, **kwargs)

#     def get_output_for(self, input, **kwargs):
#         # Use scipy.ndimage.zoom(data, (1, 2, 2)) and select bilinear
#         return input
#         # return input.sum(axis=-1)

#     def get_output_shape_for(self, input_shape):
#         return input_shape
#         # Always original dim. of image...
#         # return input_shape[:-1]


class BatchIterator_boostRegr(BatchIterator_Greedy):
    '''
    It modifies the inputs using the processInput() class.
    Optionally it modifies the targets to fit the residuals (boosting).

    Inputs:
      - processInput (None): to apply DCT or previous fixed layers. Example of input: processInput(DCT_size=4)
      - best_classifier (None): to fit the residuals
      - other usual options: batch_size, shuffle
    '''
    def __init__(self, best_classifier, objective_boost_loss, *args, **kwargs):
        self.best_classifier = best_classifier
        self.objective_boost_loss = objective_boost_loss
        super(BatchIterator_boostRegr, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):
        if yb is not None:
            # if Xb.ndim!=yb.ndim:
            #     raise ValueError('The targets are not in the right shape. \nIn order to implement a boosting classification, the fit function as targets y should get a matrix with ints [0, 0, ..., 1, ..., 0, 0] instead of just an array of integers with the class labels.')

            # Fit on residuals:
            if self.best_classifier:
                pred_proba = self.best_classifier.predict_proba(Xb) #[N,C,x,y]
                C = pred_proba.shape[1]
                pred_proba = pred_proba.transpose((0,2,3,1)).reshape((-1,C)).astype(np.float32)
                yb_vect = yb.reshape((-1,))
                prob_residuals = 1. - pred_proba[np.arange(yb_vect.shape[0]),yb_vect]
                # if self.objective_boost_loss.shape!=prob_residuals.shape:
                #     prob_residuals = np.pad(prob_residuals, ((0,self.objective_boost_loss.shape[0]-prob_residuals.shape[0])), mode='constant')
                self.objective_boost_loss.update_boosting_weights(prob_residuals)
            else:
                shape = self.objective_boost_loss.shape
                self.objective_boost_loss.update_boosting_weights(np.ones((shape,),dtype=np.float32))
        Xb, yb = super(BatchIterator_boostRegr, self).transform(Xb, yb)
        return Xb, yb

from theano.tensor.nnet import categorical_crossentropy
import theano

class categorical_crossentropy_segm_boost(object):
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
    '''
    TO BE COMPLETELY UPDATED
        - logs_path


    Options and inputs:
        - processInput (Required): istance of the class greedy_utils.processInput.
        - filter_size (7): requires an odd size to keep the same output dimension
        - best_classifier (None): best previous classifier
        - batch_size (100)
        - batchShuffle (True)
        - eval_size (0.1): decide the cross-validation proportion. Do not use the option "train_split" of NeuralNet!
        - xy_input (None, None): use only for better optimization. But then use the cloning process at your own risk.
        - all additional parameters of NeuralNet.
          (e.g. update=adam, max_epochs, update_learning_rate, etc..)

    Deprecated:
        - [num_classes now fixed to 2, that means one filter...]
        - imgShape (1024,768): used for the final residuals
        - channels_image (3): channels of the original image

    IMPORTANT REMARK: (valid only for Classification, not LogRegres...)
    in order to implement a boosting classification, the fit function as targets y should get a matrix with floats [0, 0, ..., 1, ..., 0, 0] instead of just an array of integers with the class numbers.

    To be fixed:
        - a change in the batch_size should update the batchIterator
        - add a method to update the input-processor
        - channels_input with previous fixed layers
        - check if shuffling in the batch is done in a decent way
    '''
    def __init__(self,convSoftmaxNet,**kwargs):
        info = deepcopy(kwargs)
        # --------------------------
        # Inherited by convSoftmax:
        # --------------------------
        self.xy_input = convSoftmaxNet.xy_input
        self.filter_size1 = convSoftmaxNet.filter_size1
        self.filter_size2 = convSoftmaxNet.filter_size2
        self.input_filters = convSoftmaxNet.input_filters
        self.num_filters1 = convSoftmaxNet.num_filters1
        self.num_classes = convSoftmaxNet.num_classes
        self.previous_layers = convSoftmaxNet.previous_layers
        self.eval_size = convSoftmaxNet.eval_size
        self.best_classifier = convSoftmaxNet.net if not convSoftmaxNet.net.layers_['mask'].first_iteration else None

        # -----------------
        # Own attributes:
        # -----------------
        kwargs.setdefault('name', 'boostRegr')
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
            eval_size=self.eval_size,
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
        for key in [key for key in info if 'update_' in key]:
            info[key] = info[key].get_value().item()
        json.dump(info, file(info['logs_path']+'/info-net.txt', 'w'))


    # def set_bestClassifier(self, best_classifier):
    #     '''
    #     CHECK...
    #     It updates the best_classifier. The network will then fit the
    #     residuals wrt this classifier from now on
    #     '''
    #     self.best_classifier = best_classifier
    #     customBatchIterator = BatchIterator_boostRegr(
    #         batch_size=self.batch_size,
    #         shuffle=self.batchShuffle,
    #         best_classifier=self.best_classifier,
    #         previous_layers=self.previous_layers
    #     )
    #     self.net.batch_iterator_train = customBatchIterator
    #     self.net.batch_iterator_test = customBatchIterator

    # def _reset_weights(self):
    #     '''
    #     CHECK...
    #     '''
    #     W, b = get_all_param_values(self.net.layers_['convLayer'])
    #     glorot, constant = lasagne.init.GlorotNormal(), lasagne.init.Constant()
    #     set_all_param_values(self.net.layers_['convLayer'], [glorot.sample(W.shape), constant.sample(b.shape)])

    # def clone(self,**kwargs):
    #     '''
    #     Options:
    #         - reset (False): for resetting the weights of the new Net
    #         - setClassifier (False): if set to True, the new Net will have as best_previous_classifier the previous Net

    #     IT DOESN'T HAVE ANY SENSE... CHANGE TO JUST DELETE THE WEIGHTS...

    #     Return the cloned object.
    #     '''
    #     # if self.xy_input is not (None, None):
    #     #     raise Warning("Cloning with xy-image-inputs already set...")
    #     kwargs.setdefault('reset', False)
    #     kwargs.setdefault('setClassifier', False)
    #     newObj = copy(self)
    #     if kwargs['reset']:
    #         # Reset some stuff:
    #         if newObj.net.verbose:
    #             # newObj.net.on_epoch_finished.append(PrintLog())
    #             pass
    #         newObj.net.train_history_[:] = []
    #         newObj._reset_weights()
    #     if kwargs['setClassifier']:
    #         newObj.set_bestClassifier(self.net)
    #     return newObj
