
from nolearn.lasagne import NeuralNet
from ..segmentFcts import meanIU, pixel_accuracy


# Best accuracy??
from theano.tensor import reshape
from lasagne import layers

from theano.tensor.nnet import categorical_crossentropy

def categorical_crossentropy_segm(predictions, targets):
    """Computes the categorical cross-entropy between predictions and targets.
    .. math:: L_i = - \\sum_j{t_{i,j} \\log(p_{i,j})}
    Parameters
    ----------
    predictions : Theano 2D tensor
        Predictions in (0, 1), such as softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor or 1D tensor
        Either targets in [0, 1] matching the layout of `predictions`, or
        a vector of int giving the correct class index per data point.
    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise categorical cross-entropy.
    Notes
    -----
    This is the loss function of choice for multi-class classification
    problems and softmax output units. For hard targets, i.e., targets
    that assign all of the probability to a single class per data point,
    providing a vector of int for the targets is usually slightly more
    efficient than providing a matrix with a single 1.0 per row.
    """
    shape = predictions.shape
    pred_mod = predictions.reshape((-1,shape[1]))
    if predictions.ndim == targets.ndim:
        targ_mod = targets.reshape((-1,shape[1]))
    else:
        targ_mod = targets.reshape((-1,))
    results = categorical_crossentropy(pred_mod, targ_mod)
    return results.reshape((shape[0],shape[2],shape[3]))

from theano.tensor.nnet import softmax
def softmax_segm(x):
    """Softmax activation function
    :math:`\\varphi(\\mathbf{x})_j =
    \\frac{e^{\mathbf{x}_j}}{\sum_{k=1}^K e^{\mathbf{x}_k}}`
    where :math:`K` is the total number of neurons in the layer. This
    activation function gets applied row-wise.
    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).
    Returns
    -------
    float32 where the sum of the row is 1 and each single value is in [0, 1]
        The output of the softmax function applied to the activation.
    """
    shape = x.shape
    x_mod = x.reshape((-1,shape[1]))
    results = softmax(x_mod)
    return results.reshape(shape)


class segmNeuralNet(NeuralNet):
    '''
    Modified version of NeuralNet (nolearn), adapted for a segmentation problems.

    Changes:
     - for training: store pixel acc. history
     - for validation: store pixel accuracy and meanIU history

    It won't probably work. It expects theano inputs, not arrays..
    '''
    def __init__(self,*args,**kwargs):
        kwargs['objective_loss_function'] = categorical_crossentropy_segm
        kwargs['scores_train'] = [('pixAcc_train', pixel_accuracy)]
        # kwargs['scores_valid'] = [('pixAcc_val', pixel_accuracy)]
        # kwargs['scores_valid'] = [('pixAcc', pixel_accuracy), ('meanIU', meanIU)]
        super(segmNeuralNet, self).__init__(*args, **kwargs)

