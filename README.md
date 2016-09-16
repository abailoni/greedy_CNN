# Greedy training of a CNN

The package *greedy_convnet* is an implementation of a convolutional neural network used for semantic segmentation and trained with a greedy-layer approach and a boosting procedure.

The package is based on the neural network libraries [lasagne](https://github.com/Lasagne/Lasagne) and [nolearn](https://github.com/dnouri/nolearn).

### Requirements

We recommend using [Conda](http://conda.pydata.org/docs/) (or [miniconda](http://conda.pydata.org/miniconda.html)) and [virtualenv](http://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/) for the dependencies.

To install the package, do:

```
pip install -r requirements.txt
```

It could be necessary to use conda for the missing system dependencies.

The package is compatible with the last version of [nolearn](https://github.com/dnouri/nolearn) and the last bleeding-edge versions of [theano](http://deeplearning.net/software/theano/) and [lasagne](https://github.com/Lasagne/Lasagne). All of them can be installed following the usual procedures.

## Documentation

The repository is mainly divided in two packages: *greedy_convnet* and *mod_nolearn*. In the following a brief description of main features is given.

### greedy_convnet

The package contains the implementation of the greedy network, presenting the following features:

- **Training new layers:** after initializing the main greedy network, new layers can be trained entirely or by selecting specific boosted nodes to be trained singularly
- **Backup of nodes:** after each training of a new boosted node, the full greedy network and the singular nodes are saved on disk by default. Thus the training process can be stopped at any time, without loosing almost any data
- **Restore previous models:** partially previously trained greedy networks can be restored completely or by loading singular pretrained boosted nodes (not necessarily belonging to the same greedy network)

The main class is located in file *greedy_net.py* and classes for the sub networks (boosted nodes and greedy layers) are located in folder ```greedy_convnet/sub_nets```.

### mod_nolearn

The package is an extension of the library *nolearn*, featuring:

- **collection of tools** used to perform useful actions at the end of batch iterations, epoch or training (see *nolearn_utils.py* for details)
- **grid search of hyperparameters** (*needs better integration with already existing sklearn classes*) integrated with the log, pickle and plotting tools (located in *tune_hyperparams.py*)
- specific **tools for segmentation** (located in folder ```mod_nolearn/segm/```)


## Examples of usage
Some working example scripts will be added soon.

### greedy_convnet
```python
# -------------------------------------
# Import some images from dataset:
# -------------------------------------
from various.data_utils import get_cityscapes_data

data_X_train, data_y_train, data_X_val, data_y_val = get_cityscapes_data()
X_data = (data_X_train[:480], data_X_val[:120])
y_data = (data_y_train[:480], data_y_val[:120])
X = np.concatenate(X_data)
y = np.concatenate(y_data)


# -----------------------------------------
# Some arguments of the networks:
# -----------------------------------------
from lasagne.updates import adam
main_greedyNet_kwargs = {
    'update': adam,
    'update_learning_rate': 1e-5,
    'update_beta1': 0.9,
    ...
}

greedyLayer_params = {
    'num_filters1': 10,
    'xy_input': (256, 512),
    'livePlot': True,
    ...
}

boostedNode_params = {
    'L2': 0.,
    'batch_size': 30,
    'batchShuffle': True,
    ...
}


# ----------------------------------
# Greedy subnet training functions:
# ----------------------------------
from mod_nolearn.nolearn_utils import AdjustVariable

def train_boostedNode(net):
    if net.first_node:
        net.fit(X, y, epochs=15)
    else:
        # Update some parameters:
        net.update_learning_rate.set_value(1e-5)
        net.update_AdjustObjects([
            AdjustVariable('update_beta1', start=0.9, mode='linear', stop=0.999),
            AdjustVariable('update_learning_rate', start=1e-5, mode='log', decay_rate=2e-2)
        ])
        # Train node:
        net.fit(X, y, epochs=2)
    return net


def finetune_greedyLayer(net):
    net.fit(X, y, epochs=5)
    return net

# ------ # ------ # ------- # ------- #
#          GREEDY NETWORK:            #
# ------ # ------ # ------- # ------- #

from greedy_convnet import greedyNet

# Initialize greedy network:
num_VGG16_layers = 2

greedy_network = greedyNet(
    num_VGG16_layers,
    mod='ReLU',
    model_name='greedyNet_test',
    BASE_PATH_LOG='./logs/',
    **main_greedyNet_kwargs
)


# Preload some pretrained nodes:
preLoad = {
    'boostNode_L0G0N0': ('logs/model_567742/', False), # This node is loaded
    'boostNode_L0G0N1': ('logs/model_325307/', True), # This node is loaded and trained
}
greedy_network.preLoad(preLoad)


# Train first layer:
num_nodes = 13
greedy_network.train_new_layer(
    (train_boostedNode, num_nodes, boostedNode_params),
    (finetune_greedyLayer, greedyLayer_params)
)

# Get some predictions:
pred = greedy_network.net.predict(X[0])
```


### mod_nolearn
```python
# ------ # ------ # ------- # -------  #
#        TUNE HYPERPARAMETERS:         #
# ------ # ------ # ------- # -------  #

from mod_nolearn import tuneHyperParams

# Define the training function:
class tune_lrn_rate(tuneHyperParams):
    def fit_model(self, param_values, model_name):
        # Set the hyperparameters of the model:
        model_params = {
            'path_logs': self.path_logs
            'name': model_name
            'update_learning_rate': param_values['lrn_rate']
        }

        # Train model:
        # ...
        net.fit(X, y, epochs=5)

        # Collect results:
        results = {
            'train_loss': net.train_history_[best]['train_loss'],
            'trn pixelAcc': ...
            ...
        }

        return results


# Define the tuning function:
tuning_hyperparams = tune_lrn_rate(
    (
        ('lrn_rate', 1e-7, 1e-3, 'log', np.float32),
        ('init_std', 1e-4, 1e-1, 'linear', np.float32)
    ),
    [ 'train_loss', 'valid_loss', ...], # define outputs
    num_iterations = 10,
    name = "tune_lrn_rate",
    folder_out = './tuning/',
    plot=False
)

# Fit 20 models:
tuning_hyperparams()
tuning_hyperparams()


# ----------------------------------
# Compare all losses of the 10 computed models:
# ----------------------------------
from mod_nolearn.visualize import plot_fcts_PRO, plot_stuff, get_model_data
path_tuning = "./tuning/tune_lrn_rate/"
get_data = get_model_data
xs, ys, kwargs = plot_stuff(["Training loss", "Validation loss"], path_tuning, get_data)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
plot_fcts_PRO(ax, xs, ys,log='',label_size=10,ticks_size=8,**kwargs)
fig.savefig('pdf/compare_loss.pdf')

# ----------------------------------
# Scatter plot:
# ----------------------------------
from mod_nolearn.visualize import scatter_plot
scatter_plot(path_tuning, exclude=1)

# ----------------------------------
# Plot prediction maps of the best model
# ----------------------------------
from various.utils import restore_model
model_path = './tuning/tune_lrn_rate/model_937877/model.pickle'
model = restore_model(model_path)

from mod_nolearn.visualize import plot_predict_proba
plot_predict_proba(model, X_train, y_train, "pdf/pred_maps", mistakes_plot=True, prediction_plot=True)
```
