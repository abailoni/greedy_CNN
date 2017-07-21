import sys
import yaml
import numpy as np

from lasagne.updates import adam

# import pretr_nets.vgg16 as vgg16
from greedy_convnet import greedy_net as grNet
from various.data_utils import get_cityscapes_data


sys.setrecursionlimit(20000)


# -------------------------------------
# Config file:
# -------------------------------------
configpath = 'config/config.yaml'

with open(configpath, mode='r') as configfile:
    try:
        config = yaml.load(configfile)
    except Exception as e:
        print("Could not parse YAML.")
        raise e

# To-do: add paths for each file (training and validation)
citiscape_path = config['cityscape_data']



# -------------------------------------
# Import cityscape:
# -------------------------------------

data_X_train, data_y_train, data_X_val, data_y_val = get_cityscapes_data(citiscape_path)

size_val = data_X_val.shape[0]
used_data = 1000
X_train, y_train = data_X_train[:used_data], data_y_train[:used_data]

# # Mini-images, just for test:
# CROP = 10
# data_X_train, data_y_train, data_X_val, data_y_val = data_X_train[:,:,:CROP,:CROP], data_y_train[:,:CROP,:CROP], data_X_val[:,:,:CROP,:CROP], data_y_val[:,:CROP,:CROP]


eval_size = 0.5
X_data = (X_train[:10], data_X_val[:10])
y_data = (y_train[:10], data_y_val[:10])

X = np.concatenate(X_data)
y = np.concatenate(y_data)



# --------------------------
# Fit functions for:
#
#  - finetuning greedy layer
#  - training a perceptron
#
# --------------------------

def finetune_greedyLayer(net, num_perceptron):
    net.fit(X, y, epochs=1)
    return net

def fit_perceptron(net, num_perceptron):
    if num_perceptron==0:
        net.fit(X, y, epochs=1)
    else:
        net.fit(X, y, epochs=1)
    return net

kwargs_greedyLayer = {
    'update': adam,
    'update_learning_rate': 1e-8,
    'update_beta1': 0.9}


kwargs_perceptron = {
    'update': adam,
    'update_learning_rate': 1e-8,
    'update_beta1': 0.9}



# ------------------------------
# Run greedy training on U-Net:
# ------------------------------

input_size = y_train.shape[-2:]

from various.unet_nolearn import define_network
UNET = define_network(None,
    input_size=tuple(input_size),
    depth=3,
    branching_factor=6,
    num_input_channels=3, # 3 channels image
    num_classes=2)


# Initialize greedyNet:
greedyNET = grNet.greedyNet(
    UNET,
    model_name="first_try_uNET",
    BASE_PATH_LOG="./log_nets/",
    batch_size=3,
    eval_size=eval_size,
    num_classes=2,
    verbose=1)

# If present, loads previously trained layers from log folder:
# (to-do: change and save them in a "data folder"...)
greedyNET.load_pretrained_layers()

print [key for key in greedyNET.trained_weights]


# Train another layer:
greedyNET.perform_next_greedy_step(
    fit_perceptron,
    finetune_greedyLayer,
    kwargs_perceptron=kwargs_perceptron,
    kwargs_finetune=kwargs_perceptron)

print [key for key in greedyNET.trained_weights]

# import sys
# sys.exit("Error message")

# Another one:
greedyNET.perform_next_greedy_step(
    fit_perceptron,
    finetune_greedyLayer,
    kwargs_perceptron=kwargs_perceptron,
    kwargs_finetune=kwargs_perceptron)



