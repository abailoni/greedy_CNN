import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------
# Import CIFAR10:
# -------------------------------------
from cs231n.data_utils import get_CIFAR10_data

# Load the (preprocessed) CIFAR10 data.
data = get_CIFAR10_data()
for k, v in data.iteritems():
    print '%s: ' % k, v.shape

# Use one third of the data for the moment:
used_data = 200
# nolearn divide by itself the data in training and validation:
X, y = data['X_train'][:used_data], data['y_train'][:used_data]

# Don't forget to use right data types!
# To convert the inputs (they are float64 and int64) use
X, y = X.astype(np.float32), y.astype(np.int32)
print y.dtype
# -------------------------------------
# Design the network with nolearn
# -------------------------------------
from lasagne import layers as lasLayers
from lasagne.updates import adam
from lasagne.nonlinearities import sigmoid
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit

# CONSTANTS:
num_classes = 2
filter_size = 8


# CONSTRUCT NEW LAYERS:

class UpScaleLayer(lasLayers.Layer):
    def get_output_for(self, input, **kwargs):
        # Use scipy.ndimage.zoom(data, (1, 2, 2)) and select bilinear
        return input.sum(axis=-1)


    def get_output_shape_for(self, input_shape):
        return input_shape[:-1]



layers = [
    # layer dealing with the input data
    (lasLayers.InputLayer, {'shape': (None, None, None, None)}),

    # first stage of our convolutional layers
    (lasLayers.Conv2DLayer, {'num_filters': num_classes, 'filter_size': filter_size, 'pad':1, 'nonlinearity': sigmoid}),
    (lasLayers.TransposedConv2DLayer, {'num_filters': num_classes, 'filter_size': filter_size}),
]

net2 = NeuralNet(layers=layers, update=adam, update_learning_rate=0.01,update_beta1=0.9,train_split=TrainSplit(eval_size=0.1),regression=False,max_epochs=10,verbose=1)

# Here we compile:
net2.initialize()

# -------------------------------------
# Visualize some useful stuff:
# -------------------------------------
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion
from nolearn.lasagne.visualize import plot_saliency

# Print network in a pdf:
from nolearn.lasagne.visualize import draw_to_file
draw_to_file(net2,"prova_graph.pdf")

# Really useful information about the health status of the net:
from nolearn.lasagne import PrintLayerInfo
layer_info = PrintLayerInfo()
net2.verbose=2
layer_info(net2)



# -------------------------------------
# Start training: (Dominic, I hate you...)
# -------------------------------------
net2.max_epochs = 10

# quit()
net2.fit(X, y)
