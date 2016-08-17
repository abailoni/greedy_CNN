import matplotlib.pyplot as plt

def plot_loss(net, name_file="loss.pdf"):
    fig = plt.figure()
    ax=fig.add_subplot(111)
    train_loss = [row['train_loss'] for row in net.train_history_]
    valid_loss = [row['valid_loss'] for row in net.train_history_]
    ax.plot(train_loss, label='train loss')
    ax.plot(valid_loss, label='valid loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend(loc='best')
    fig.savefig(name_file)

from itertools import product
import numpy as np

def plot_conv_weights(layer, figsize=(6, 6)):
    """Plot the weights of a specific layer.
    Only really makes sense with convolutional layers.
    Parameters
    ----------
    layer : lasagne.layers.Layer
    """
    W = layer.W.get_value()
    shape = W.shape
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows

    figs, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
        if i >= shape[0]:
            break
        axes[r, c].imshow(W[i].transpose((1,2,0)), interpolation='nearest')
    return plt


def plot_images(images, figsize=(6, 6)):
    """Plot the weights of a specific layer.
    Only really makes sense with convolutional layers.
    Parameters
    ----------
    layer : N,channels,x,y
    """
    shape = images.shape
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows

    figs, axes = plt.subplots(nrows, ncols, squeeze=False)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
        if i >= shape[0]:
            break
        axes[r, c].imshow(images[i].transpose((1,2,0)), interpolation='nearest')
    return plt

# def plot_images(images):
#     N,
#     W = layer.W.get_value()
#     shape = images.shape
#     nrows = np.ceil(np.sqrt(shape[0])).astype(int)
#     ncols = nrows

#     for feature_map in range(shape[1]):
#         figs, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

#         for ax in axes.flatten():
#             ax.set_xticks([])
#             ax.set_yticks([])
#             ax.axis('off')

#         for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
#             if i >= shape[0]:
#                 break
#             axes[r, c].imshow(W[i, feature_map], cmap='gray',
#                               interpolation='none')
#     return plt
