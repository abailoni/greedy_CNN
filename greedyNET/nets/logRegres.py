from lasagne import layers
from lasagne.nonlinearities import softmax

from lasagne.layers import set_all_param_values, get_all_param_values
import lasagne.init

from ...mod_nolearn.nets import segmNeuralNet
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
    '''
    def __init__(self, *args, **kwargs):
        self.imgShape = kwargs.pop('imgShape', DEFAULT_imgShape)
        super(UpScaleLayer, self).__init__(*args, **kwargs)

    def get_output_for(self, input, **kwargs):
        # Use scipy.ndimage.zoom(data, (1, 2, 2)) and select bilinear

        return input.sum(axis=-1)

    def get_output_shape_for(self, input_shape):
        # Always original dim. of image...
        return input_shape[:-1]

class LogRegr(object):
    '''
    Inputs:
        - filter_size (8)
        - num_classes (2)
        - imgShape (1024,768)
        - best_classifier: best previous classifier (identity network)
        - all additional parameters of NeuralNet
          (e.g. update=adam, max_epochs, update_learning_rate, etc..)
    '''
    def __init__(self,**kwargs):
        self.filter_size = kwargs.pop('filter_size', 8)
        self.num_classes = kwargs.pop('num_classes', 2)
        self.imgShape = kwargs.pop('imgShape', DEFAULT_imgShape)
        # Wrong, should follow the idea of net.eval(X):
        self.best_classifier = kwargs.pop('best_classifier', lambda X: X)

        netLayers = [
            # layer dealing with the input data
            (layers.InputLayer, {'shape': (None, None, None, None)}),

            # first stage of our convolutional layers
            (layers.Conv2DLayer, {'name': 'convLayer', 'num_filters': self.num_classes, 'filter_size': self.filter_size, 'pad':1, 'nonlinearity': softmax}),
            (UpScaleLayer, {'imgShape':self.imgShape}),
        ]

        self.net = segmNeuralNet(layers=netLayers, **kwargs)
        print "Compiling network..."
        self.net.initialize()
        print "Done"

    def set_bestClassifier(self, best_classifier):
        self.best_classifier = best_classifier

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








