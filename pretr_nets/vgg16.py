# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

from lasagne.layers import InputLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
# from lasagne.layers import Conv2DLayer as ConvLayer

import pickle
from lasagne.layers import set_all_param_values
from lasagne.layers import get_output
import theano.tensor as T
from theano import function

def build_model(input_var, data_size=(None, 3, None, None)):
    net = {}
    net['input'] = InputLayer(data_size, input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    # net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    # net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    # net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    # net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    # net['fc8'] = DenseLayer(
    #     net['fc7_dropout'], num_units=1000, nonlinearity=None)
    # net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net

# IMPORT WEIGHTS:

def import_model():
    print "Importing pretrained net..."
    model = pickle.load(open('/mnt/localdata01/abailoni/pretrained/vgg16.pkl'))
    return model


# def import_pretr_vgg16_layers():
#     '''
#     Should be generalized for a general number of kept layers...
#     '''
#     model = import_model()
#     # CLASSES = model['synset words']
#     # MEAN_IMAGE = model['mean image']

#     input_var = T.tensor4('input')
#     net = build_model(input_var)
#     output_layer = net['conv2_2']

#     print "Loading pretrained weights..."
#     set_all_param_values(output_layer, model['param values'][:8])
#     print "Done!"

#     prediction = get_output(output_layer, deterministic=True)
#     eval_fun_vgg16 = function([input_var], prediction, name='eval_fun_vgg16')
#     print "Compiled!"

#     return eval_fun_vgg16, output_layer


def nolearn_vgg16_layers(data_size=(None, 3, None, None)):
    layers = [
        (InputLayer, {
            'shape': data_size,
            }),
        (ConvLayer, {
            'name': 'conv1_1',
            'num_filters': 64,
            'filter_size': 3,
            'pad': 1,
            'flip_filters': False
            }),
        (ConvLayer, {
            'name': 'conv1_2',
            'num_filters': 64,
            'filter_size': 3,
            'pad': 1,
            'flip_filters': False
            }),
        (PoolLayer, {
            'pool_size': 2,
            }),
        (ConvLayer, {
            'name': 'conv2_1',
            'num_filters': 128,
            'filter_size': 3,
            'pad': 1,
            'flip_filters': False
            }),
        (ConvLayer, {
            'name': 'conv2_2',
            'num_filters': 128,
            'filter_size': 3,
            'pad': 1,
            'flip_filters': False
            }),
    ]
    return layers


def nolearn_insert_weights_vgg16(net, num_vgg_layers):
    model = import_model()
    layer_name = nolearn_vgg16_layers()[num_vgg_layers][1]['name']
    set_all_param_values(net.layers_[layer_name], model['param values'][:num_vgg_layers*2])
    return net

