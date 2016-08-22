import time
import numpy as np
from copy import deepcopy

from lasagne import layers
from lasagne.layers import set_all_param_values, get_all_param_values
import lasagne.init
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


class BatchIterator_BoostLogRegr(greedy_utils.BatchIterator_Greedy):
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
        super(BatchIterator_BoostLogRegr, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):
        Xb, yb = super(BatchIterator_BoostLogRegr, self).transform(Xb, yb)

        if yb is not None:
            # if Xb.ndim!=yb.ndim:
            #     raise ValueError('The targets are not in the right shape. \nIn order to implement a boosting classification, the fit function as targets y should get a matrix with ints [0, 0, ..., 1, ..., 0, 0] instead of just an array of integers with the class labels.')

            # Fit on residuals:
            yb = yb.astype(np.float32)
            if self.best_classifier:
                pred = self.best_classifier.predict_proba(Xb).squeeze() #[N,x,y]
                yb = np.absolute(pred - yb)
        return Xb, yb


class Boost_LogRegr(object):
    '''
    Inputs:
        - filter_size (7): requires an odd size to keep the same output dimension
        - [num_classes now fixed to 2, that means one filter...]
        - imgShape (1024,768): used for the final residuals
        - channels_image (3): channels of the original image
        - xy_input (None, None): if not set then the network can be reused for different inputs
        - best_classifier (None): best previous classifier
        - batch_size (100)
        - eval_size (0.1): decide the cross-validation proportion. Do not use the option "train_split" of NeuralNet!
        - DCT_size (None): if set to None the channels of the input image will be used
        - fixed_previous_layers (None): if this is set, DCT_size is ignored (and channels_image is redundant)
        - all additional parameters of NeuralNet.
          (e.g. update=adam, max_epochs, update_learning_rate, etc..)

    IMPORTANT REMARK: (valid only for Classification, not LogRegres...)
    in order to implement a boosting classification, the fit function as targets y should get a matrix with floats [0, 0, ..., 1, ..., 0, 0] instead of just an array of integers with the class numbers.

    To be fixed:
        - a change in the batch_size should update the batchIterator
        - add a method to update the input-processor
        - channels_input with previous fixed layers
        - check if shuffling in the batch is done in a decent way
    '''
    def __init__(self,**kwargs):
        self.filter_size = kwargs.pop('filter_size', 7)
        self.num_classes = 1
        if "num_classes" in kwargs:
            raise Warning('No idea how to implement a classification boosting for the moment. "num_classes" parameter is fixed to 2')
        self.batch_size = kwargs.pop('batch_size', 100)
        self.channels_image = kwargs.pop('channels_image', 3)
        self.xy_input = kwargs.pop('xy_input', (None, None))
        self.imgShape = kwargs.pop('imgShape', DEFAULT_imgShape)
        self.best_classifier = kwargs.pop('best_classifier', False)
        self.eval_size = kwargs.pop('eval_size', 0.1)
        if "train_split" in kwargs:
            raise ValueError('The option train_split is not used. Use eval_size instead.')

        # Input processing:
        self.fixed_previous_layers = kwargs.pop('fixed_previous_layers', None)
        self.DCT_size = kwargs.pop('DCT_size', None)
        if self.fixed_previous_layers:
            if 'DCT_size':
                raise Warning('DCT_size ignored. Using output of previous fixed layers instead.')

        self.batchShuffle = kwargs.pop('batchShuffle', True)
        self.processInput = greedy_utils.processInput(
                DCT_size=self.DCT_size,
                fixed_layers=self.fixed_previous_layers
            )
        customBatchIterator = BatchIterator_BoostLogRegr(
            batch_size=self.batch_size,
            shuffle=self.batchShuffle,
            best_classifier=self.best_classifier,
            processInput=self.processInput
        )

        # Layers:
        netLayers = [
            # layer dealing with the input data
            (layers.InputLayer, {'shape': (None, self.processInput.output_channels, self.xy_input[0], self.xy_input[1])}),
            # first stage of our convolutional layers
            (layers.Conv2DLayer, {
                'name': 'convLayer',
                'num_filters': self.num_classes,
                'filter_size': self.filter_size,
                'pad':'same',
                'nonlinearity': segmNet.sigmoid_segm}),
            # (UpScaleLayer, {'imgShape':self.imgShape}),
        ]

        self.net = segmNet.segmNeuralNet(
            layers=netLayers,
            batch_iterator_train = customBatchIterator,
            batch_iterator_test = customBatchIterator,
            objective_loss_function = segmNet.binary_crossentropy_segm,
            scores_train = [('trn pixelAcc', pixel_accuracy_sigmoid)],
            scores_valid = [('val pixelAcc', pixel_accuracy_sigmoid)],
            y_tensor_type = T.ftensor3,
            eval_size=self.eval_size,
            regression = True,
            **kwargs
        )
        print "\n\n---------------------------\nCompiling LogRegr network...\n---------------------------"
        tick = time.time()
        self.net.initialize()
        tock = time.time()
        print "Done! (%f sec.)\n\n\n" %(tock-tick)

    def set_bestClassifier(self, best_classifier):
        '''
        It updates the best_classifier. The network will then fit the
        residuals wrt this classifier from now on
        '''
        self.best_classifier = best_classifier
        customBatchIterator = BatchIterator_BoostLogRegr(
            batch_size=self.batch_size,
            shuffle=self.batchShuffle,
            best_classifier=self.best_classifier,
            processInput=self.processInput
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

        Return the cloned object.
        '''
        kwargs.setdefault('reset', False)
        kwargs.setdefault('setClassifier', False)
        newObj = deepcopy(self)
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
