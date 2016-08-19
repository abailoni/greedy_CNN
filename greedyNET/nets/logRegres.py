import time
import numpy as np
from copy import deepcopy

from scipy.fftpack import dct

from lasagne import layers
from lasagne.layers import set_all_param_values, get_all_param_values
import lasagne.init
from mod_nolearn.nets.segmNet import segmNeuralNet, softmax_segm


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

from nolearn.lasagne import BatchIterator
import theano.tensor as T

class segmBatchIterator(BatchIterator):
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
        self.processInput = kwargs.pop('processInput', None)
        super(segmBatchIterator, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):
        # Process inputs:
        if self.processInput:
            Xb = self.processInput(Xb)

        # This actually is true only for the logistic regression...
        if yb is not None:
            if Xb.ndim!=yb.ndim:
                raise ValueError('The targets are not in the right shape. \nIn order to implement a boosting classification, the fit function as targets y should get a matrix with ints [0, 0, ..., 1, ..., 0, 0] instead of just an array of integers with the class labels.')

            # Fit on residuals:
            if self.best_classifier:
                pred = self.best_classifier.predict_proba(Xb) #[N,C,x,y]
                print pred[0,:,0,0]
                print yb[0,:,0,0]
                yb = np.absolute(pred - yb)
                print yb[0,:,0,0]
                raise ValueError("Stop here")
            else:
                # This is not necessary in Net2 where we fit always the GrTruth
                # because y_tensor_type can be set to T.itensor4
                yb = yb.astype(np.float32)

        return Xb, yb


class LogRegr(object):
    '''
    Inputs:
        - filter_size (7): requires an odd size to keep the same output dimension
        - num_classes (2)
        - imgShape (1024,768): used for the final residuals
        - channels_image (3): channels of the original image
        - xy_input (None, None): if not set then the network can be reused for different inputs
        - best_classifier (None): best previous classifier
        - batch_size (100)
        - eval_size (0.1): decide the cross-validation proportion. Do not use the option "train_split" of NeuralNet!
        - DCT_size (7): if set to None the channels of the input image will be used
        - fixed_previous_layers (None): if this is set, DCT_size is ignored (and channels_image is redundant)
        - all additional parameters of NeuralNet.
          (e.g. update=adam, max_epochs, update_learning_rate, etc..)

    IMPORTANT REMARK:
    in order to implement a boosting classification, the fit function as targets y should get a matrix with floats [0, 0, ..., 1, ..., 0, 0] instead of just an array of integers with the class numbers.

    To be fixed:
        - a change in the batch_size should update the batchIterator
        - add a method to update the input-processor
        - channels_input with previous fixed layers
        - check if shuffling in the batch is done in a decent way
    '''
    def __init__(self,**kwargs):
        self.filter_size = kwargs.pop('filter_size', 7)
        self.num_classes = kwargs.pop('num_classes', 2)
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
        if self.fixed_previous_layers:
            # self.channels_input = ???
            if 'DCT_size' in kwargs:
                raise Warning('DCT_size ignored. Using output of previous fixed layers instead.')
        else:
            self.DCT_size = kwargs.pop('DCT_size', 7)
            self.channels_input = self.channels_image * self.DCT_size**2 if self.DCT_size else self.channels_image

        self.batchShuffle = True
        customBatchIterator = segmBatchIterator(
            batch_size=self.batch_size,
            shuffle=self.batchShuffle,
            best_classifier=self.best_classifier,
            processInput=processInput(
                DCT_size=self.DCT_size,
                fixed_layers=self.fixed_previous_layers
            )
        )

        # Layers:
        netLayers = [
            # layer dealing with the input data
            (layers.InputLayer, {'shape': (None, self.channels_input, self.xy_input[0], self.xy_input[1])}),

            # first stage of our convolutional layers
            (layers.Conv2DLayer, {'name': 'convLayer', 'num_filters': self.num_classes, 'filter_size': self.filter_size, 'pad':'same', 'nonlinearity': softmax_segm}),
            # (UpScaleLayer, {'imgShape':self.imgShape}),
        ]

        self.net = segmNeuralNet(layers=netLayers,
            batch_iterator_train = customBatchIterator,
            batch_iterator_test = customBatchIterator,
            y_tensor_type = T.ftensor4,
            eval_size=self.eval_size,
            **kwargs
        )
        print "\n\n---------------------------\nCompiling LogRegr network...\n---------------------------"
        tick = time.time()
        self.net.initialize()
        tock = time.time()
        print "Done! (%f sec.)\n\n\n" %(tock-tick)

    def set_bestClassifier(self, best_classifier):
        ''' It updates the best_classifier. The network will then fit the
        residuals wrt this classifier from now on.'''
        self.best_classifier = best_classifier
        customBatchIterator = segmBatchIterator(
            batch_size=self.batch_size,
            shuffle=self.batchShuffle,
            best_classifier=self.best_classifier,
            processInput=processInput(
                DCT_size=self.DCT_size,
                fixed_layers=self.fixed_previous_layers
            )
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
        '''
        kwargs.setdefault('reset', False)
        kwargs.setdefault('setClassifier', False)
        newObj = deepcopy(self)
        newObj.train_history_ = []
        if kwargs['reset']:
            newObj._reset_weights()
        if kwargs['setClassifier']:
            newObj.set_bestClassifier(self.net)
        return newObj


### INPUT FUNCTION:

class processInput(object):
    '''
    Compute the input. The function is called each time we choose a batch of data.

    Arguments:
     - DCT_size (7): size of the DCT filter, the output will have 7*7*3 filters. The input can be set to None and then nothing will be done.
     - fixed_layers (None): optional network representing the previous learned and fixed layers (the DCT filters should be included...?)
    '''
    def __init__(self, **kwargs):
        self.DCT_size = kwargs.pop('DCT_size', 7)
        self.fixed_layers = kwargs.pop('fixed_layers', None)
        if kwargs:
            raise Warning('Additional not necesary arguments to handleInput have been passed')

    def __call__(self, batch_input):
        if self.fixed_layers!=None:
            batch_output = self.fixed_layers.predict_proba(batch_input)
        elif self.DCT_size:
            batch_output = self.apply_DCT(batch_input)
        else:
            batch_output = batch_input

        return batch_output

    def apply_DCT(self, batch_input):
        '''
        Size of the batch_input: (N,3,dim_x,dim_y)
        It needs to be implemented at least in cython... Or..?
        '''
        N, channels, dim_x, dim_y = batch_input.shape
        pad = self.DCT_size/2

        temp = np.empty((N,channels*self.DCT_size,dim_x,dim_y+2*pad)) #dct along one dim.
        output = np.empty((N,channels*self.DCT_size**2,dim_x,dim_y), dtype=np.float32)
        padded_input = np.pad(batch_input,pad_width=((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')

        tick = time.time()
        for i in range(dim_x):
            # Note that (i,j) are the center of the filter in input, but are the top-left corner coord. in the padded_input
            if i%10==0:
                print i
            temp[:,:,i,:] = np.reshape(dct(padded_input[:,:,i:,:], axis=2, n=self.DCT_size), (N,-1,dim_y+2*pad))
        for j in range(dim_y):
            if j%10==0:
                print j
            output[:,:,:,j] = dct(temp[:,:,:,j:], axis=3, n=self.DCT_size).reshape((N,-1,dim_x)).astype(np.float32)
        tock = time.time()
        print "Conversion: %g sec." %(tock-tick)
        return output


# --------------------------
# NETWORK 2:
# --------------------------

class Network2(object):
    '''
    This class contains the network that put together the LogRegr networks computed in a boosting procedure.

    Inputs:
        - ...
    '''
    def __init__(self):
        pass







