# Greedy training of a CNN

The package *greedy_convnet* is an implementation of a convolutional neural network used for semantic segmentation and trained with a greedy-layer approach and a boosting procedure.

The package is based on the neural network libraries [lasagne](https://github.com/Lasagne/Lasagne) and [nolearn](https://github.com/dnouri/nolearn).

### Requirements

We recommend using [Conda](http://conda.pydata.org/docs/) (or [miniconda](http://conda.pydata.org/miniconda.html)) and [virtualenv](http://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/) for the dependencies.

To install the package, do:

```
pip install -r https://github.com/abailoni/greedy_CNN/requirements.txt
```

It could be necessary to use conda for the missing system dependencies.

The package is compatible with the last version of [nolearn](https://github.com/dnouri/nolearn) and the last bleeding-edge versions of [theano](http://deeplearning.net/software/theano/) and [lasagne](https://github.com/Lasagne/Lasagne). All of them can be installed following the usual procedures.

## Documentation

The repository is mainly divided in two packages: *greedy_convnet* and *mod_nolearn*. In the following a brief documentation of them is given.

### greedy_convnet

The packages contains the implementation of the greedy network, presenting the following features:

- **Training new layers:** after initializing the main greedy network, new layers can be trained entirely or by selecting specific boosted nodes to be trained singularly
- **Backup of nodes:** after each training of a new boosted node, the full greedy network and the singular nodes are saved on disk by default. Thus the training process can be stopped at any time, without loosing almost any data
- **Restore previous models:** partially previously trained greedy networks can be restored completely or by loading singular pretrained boosted nodes (not necessarily belonging to the same greedy network)

The main class is located in file *greedy_net.py* and classes for the sub networks (boosted nodes and greedy layers) are located in folder ```greedy_convnet/sub_nets```.

##### Example

```python
print "Ciao"
```

### mod_nolearn

The package is an extension of the library *nolearn*, featuring:

- **collection of tools** used to perform useful actions at the end of batch iterations, epoch or training (see *nolearn_utils.py* for details)
- **grid search of hyperparameters** (*needs better integration with already existing sklearn classes*) integrated with the log, pickle and plotting tools (located in *tune_hyperparams.py*)
- specific **tools for segmentation** (located in folder ```mod_nolearn/segm/```)

##### Example
