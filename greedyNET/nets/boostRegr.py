import time
import numpy as np
from copy import deepcopy, copy

from lasagne import layers
from lasagne.layers import set_all_param_values, get_all_param_values
import lasagne.init
from lasagne.nonlinearities import identity, rectify
from lasagne.objectives import squared_error
import theano.tensor as T
from nolearn.lasagne import PrintLog

import mod_nolearn.nets.segmNet as segmNet
import greedyNET.greedy_utils as greedy_utils
from mod_nolearn.segmentFcts import pixel_accuracy_sigmoid

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


class BatchIterator_boostRegr(greedy_utils.BatchIterator_Greedy):
    '''
    It modifies the inputs using the processInput() class.
    Optionally it modifies the targets to fit the residuals (boosting).

    Inputs:
      - processInput (None): to apply DCT or previous fixed layers. Example of input: processInput(DCT_size=4)
      - best_classifier (None): to fit the residuals
      - other usual options: batch_size, shuffle
    '''
    def __init__(self, *args, **kwargs):
        self.best_classifier = kwargs.pop('best_classifier', None)
        super(BatchIterator_boostRegr, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):
        if yb is not None:
            # if Xb.ndim!=yb.ndim:
            #     raise ValueError('The targets are not in the right shape. \nIn order to implement a boosting classification, the fit function as targets y should get a matrix with ints [0, 0, ..., 1, ..., 0, 0] instead of just an array of integers with the class labels.')

            # Fit on residuals:
            yb = yb.astype(np.float32)
            if self.best_classifier:
                pred = self.best_classifier.predict_proba(Xb).squeeze() #[N,x,y]
                yb = pred - yb

        Xb, yb = super(BatchIterator_boostRegr, self).transform(Xb, yb)
        return Xb, yb


class boostRegr_routine(object):
    '''
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
    def __init__(self,previous_layers,input_filters,best_classifier=None,**kwargs):
        # -----------------
        # Own attributes:
        # -----------------
        self.filter_size1 = kwargs.pop('filter_size1', 7)
        self.filter_size2 = kwargs.pop('filter_size2', 7)
        self.num_filters1 = kwargs.pop('num_filters1', 5)
        self.input_filters = input_filters
        self.num_classes = 1
        if "num_classes" in kwargs:
            raise Warning('Multy-class classification boosting not implemented for the moment. "num_classes" parameter is fixed to 2')
        self.batch_size = kwargs.pop('batch_size', 100)
        self.xy_input = kwargs.pop('xy_input', (None, None))
        self.best_classifier = kwargs.pop('best_classifier', None)
        if not self.best_classifier:
            self.best_classifier = best_classifier
        self.eval_size = kwargs.pop('eval_size', 0.1)
        if "train_split" in kwargs:
            raise ValueError('The option train_split is not used. Use eval_size instead.')
        # self.channels_image = kwargs.pop('channels_image', 3)
        # self.imgShape = kwargs.pop('imgShape', DEFAULT_imgShape)


        # -----------------
        # Input processing:
        # -----------------
        self.batchShuffle = kwargs.pop('batchShuffle', True)
        self.previous_layers = previous_layers
        customBatchIterator = BatchIterator_boostRegr(
            batch_size=self.batch_size,
            shuffle=self.batchShuffle,
            best_classifier=self.best_classifier,
            previous_layers=self.previous_layers
        )

        # -----------------
        # Building NET:
        # -----------------
        # Check if it's a regression or the first softmax:
        if self.best_classifier:
            final_nonlinearity = identity
            objective_loss_function = squared_error
        else:
            # Change here for a multiclass softmax:
            final_nonlinearity = segmNet.sigmoid_segm
            objective_loss_function = segmNet.binary_crossentropy_segm
            # HERE INSERT POSSIBLE WEIGHTS OF PREVIOUS LEVEL:
            pass
        netLayers = [
            # layer dealing with the input data
            (layers.InputLayer, {'shape': (None, input_filters, self.xy_input[0], self.xy_input[1])}),
            (layers.Conv2DLayer, {
                'name': 'conv1',
                'num_filters': self.num_filters1,
                'filter_size': self.filter_size1,
                'pad':'same',
                'nonlinearity': rectify}),
            (layers.Conv2DLayer, {
                'name': 'conv2',
                'num_filters': self.num_classes,
                'filter_size': self.filter_size1,
                'pad':'same',
                'nonlinearity': final_nonlinearity}),
            # (UpScaleLayer, {'imgShape':self.imgShape}),
        ]

        self.net = segmNet.segmNeuralNet(
            layers=netLayers,
            batch_iterator_train = customBatchIterator,
            batch_iterator_test = customBatchIterator,
            objective_loss_function = objective_loss_function,
            y_tensor_type = T.ftensor3,
            eval_size=self.eval_size,
            regression = True,
            **kwargs
        )
        # print "\n\n---------------------------\nCompiling regr network...\n---------------------------"
        # tick = time.time()
        self.net.initialize()
        # tock = time.time()
        # print "Done! (%f sec.)\n\n\n" %(tock-tick)

    def set_bestClassifier(self, best_classifier):
        '''
        It updates the best_classifier. The network will then fit the
        residuals wrt this classifier from now on
        '''
        self.best_classifier = best_classifier
        customBatchIterator = BatchIterator_boostRegr(
            batch_size=self.batch_size,
            shuffle=self.batchShuffle,
            best_classifier=self.best_classifier,
            previous_layers=self.previous_layers
        )
        self.net.batch_iterator_train = customBatchIterator
        self.net.batch_iterator_test = customBatchIterator

    def _reset_weights(self):
        W, b = get_all_param_values(self.net.layers_['convLayer'])
        glorot, constant = lasagne.init.GlorotNormal(), lasagne.init.Constant()
        set_all_param_values(self.net.layers_['convLayer'], [glorot.sample(W.shape), constant.sample(b.shape)])

    def clone(self,**kwargs):
        '''
        Options:
            - reset (False): for resetting the weights of the new Net
            - setClassifier (False): if set to True, the new Net will have as best_previous_classifier the previous Net

        IT DOESN'T HAVE ANY SENSE... CHANGE TO JUST DELETE THE WEIGHTS...

        Return the cloned object.
        '''
        # if self.xy_input is not (None, None):
        #     raise Warning("Cloning with xy-image-inputs already set...")
        kwargs.setdefault('reset', False)
        kwargs.setdefault('setClassifier', False)
        newObj = copy(self)
        if kwargs['reset']:
            # Reset some stuff:
            if newObj.net.verbose:
                # newObj.net.on_epoch_finished.append(PrintLog())
                pass
            newObj.net.train_history_[:] = []
            newObj._reset_weights()
        if kwargs['setClassifier']:
            newObj.set_bestClassifier(self.net)
        return newObj
