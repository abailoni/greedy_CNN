import time

from lasagne import layers

from lasagne.layers import set_all_param_values, get_all_param_values
import lasagne.init

from mod_nolearn.nets.segmNet import segmNeuralNet, softmax_segm
DEFAULT_imgShape = (1024,768)

class UpScaleLayer(layers.Layer):
    '''
    --------------------------
    Subclass of lasagne.layers.Layer:
    --------------------------

    Upscale (if necessary) the output of the logistic regression to match the
    dimensions of the initial image.

    The algorithm used is bilinear interpolation.
    The layer is not Trainable at this stage.

    UNCOMPLETE: for the moment is an identity layer.
    '''
    def __init__(self, *args, **kwargs):
        self.imgShape = kwargs.pop('imgShape', DEFAULT_imgShape)
        super(UpScaleLayer, self).__init__(*args, **kwargs)

    def get_output_for(self, input, **kwargs):
        # Use scipy.ndimage.zoom(data, (1, 2, 2)) and select bilinear
        return input
        # return input.sum(axis=-1)

    def get_output_shape_for(self, input_shape):
        return input_shape
        # Always original dim. of image...
        # return input_shape[:-1]

from nolearn.lasagne import BatchIterator
import theano.tensor as T
from theano.tensor import abs_

class fit_residuals(BatchIterator):
    def __init__(self, *args, **kwargs):
        self.best_classifier = kwargs.pop('best_classifier', False)
        super(fit_residuals, self).__init__(*args, **kwargs)


    def transform(self, Xb, yb):
        # Not working for some reason...
        if Xb.ndim!=yb.ndim:
            raise ValueError('The targets are not in the right shape. \nIn order to implement a boosting classification, the fit function as targets y should get a matrix with floats [0, 0, ..., 1, ..., 0, 0] instead of just an array of integers with the class numbers.')

        # Fit on residuals:
        if self.best_classifier:
            pred = self.best_classifier.predict(Xb) #[N,C,x,y]
            yb = abs_(pred - yb)
            print "Ciao"

        return Xb, yb


class LogRegr(object):
    '''
    Inputs:
        - filter_size (8)
        - num_classes (2)
        - imgShape (1024,768): used for the final residuals
        - channels_input (64): can not be set to 'None'
        - xy_input (None, None): if not set then the network can be reused for different inputs
        - best_classifier (None): best previous classifier
        - batch_size (100)
        - all additional parameters of NeuralNet
          (e.g. update=adam, max_epochs, update_learning_rate, etc..)

    IMPORTANT REMARK:
    in order to implement a boosting classification, the fit function as targets y should get a matrix with floats [0, 0, ..., 1, ..., 0, 0] instead of just an array of integers with the class numbers.

    To be fixed:
        - Padding is not automatic..
        - a change in the batch_size should update the batchIterator
    '''
    def __init__(self,**kwargs):
        self.filter_size = kwargs.pop('filter_size', 7)
        self.num_classes = kwargs.pop('num_classes', 2)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.channels_input = kwargs.pop('channels_input', 64)
        self.xy_input = kwargs.pop('xy_input', (None, None))
        self.imgShape = kwargs.pop('imgShape', DEFAULT_imgShape)
        self.best_classifier = kwargs.pop('best_classifier', False)

        netLayers = [
            # layer dealing with the input data
            (layers.InputLayer, {'shape': (None, self.channels_input, self.xy_input[0], self.xy_input[1])}),

            # first stage of our convolutional layers
            (layers.Conv2DLayer, {'name': 'convLayer', 'num_filters': self.num_classes, 'filter_size': self.filter_size, 'pad':3, 'nonlinearity': softmax_segm}),
            (UpScaleLayer, {'imgShape':self.imgShape}),
        ]

        self.net = segmNeuralNet(layers=netLayers,
            batch_iterator_train = fit_residuals(batch_size=self.batch_size,best_classifier=self.best_classifier),
            batch_iterator_test = fit_residuals(batch_size=self.batch_size,best_classifier=self.best_classifier),
            y_tensor_type = T.itensor4,
            **kwargs
        )
        print "\n\n---------------------------\nCompiling network...\n---------------------------"
        tick = time.time()
        self.net.initialize()
        tock = time.time()
        print "Done! (%f sec.)" %(tock-tick)

    def set_bestClassifier(self, best_classifier):
        self.best_classifier = best_classifier
        self.net.batch_iterator_train = fit_residuals(batch_size=self.batch_size,best_classifier=self.best_classifier)
        self.net.batch_iterator_test = fit_residuals(batch_size=self.batch_size,best_classifier=self.best_classifier)


    def _reset_weights(self):
        W, b = get_all_param_values(self.net.layers_['convLayer'])
        glorot, constant = lasagne.init.GlorotNormal(), lasagne.init.Constant()
        set_all_param_values(self.net.layers_['convLayer'], [glorot.sample(W.shape), constant.sample(b.shape)])

    def clone(self,**kwargs):
        '''
        Accept option 'reset' for resetting the weights of the new Net
        '''
        kwargs.setdefault('reset', False)
        newObj = self.deepcopy()
        if kwargs['reset']:
            newObj._reset_weights()
        return newObj









