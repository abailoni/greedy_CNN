import numpy as np


from mod_nolearn.nets.modNeuralNet import modNeuralNet
from ..segmentFcts import meanIU, pixel_accuracy


# Best accuracy??
from theano.tensor import reshape

from theano.tensor.nnet import categorical_crossentropy


# This is computing the loss:
def categorical_crossentropy_segm(predictions, targets):
    """
    MODIFICATIONS:
        - reshape targets to get a good match

    Computes the categorical cross-entropy between predictions and targets.
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
    pred_mod1 = predictions.transpose((0,2,3,1))
    pred_mod = pred_mod1.reshape((-1,shape[1]))
    if predictions.ndim == targets.ndim:
        targ_mod1 = targets.transpose((0,2,3,1))
        targ_mod = targ_mod1.reshape((-1,shape[1]))
    else:
        targ_mod = targets.reshape((-1,))
    results = 1./(shape[0]) * categorical_crossentropy(pred_mod, targ_mod)
    return results.reshape((shape[0],shape[2],shape[3]))

# This is the non-linearity, giving the scores:
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
    x_mod = x.transpose((0,2,3,1)).reshape((-1,shape[1]))
    results = softmax(x_mod)
    return results.reshape((shape[0],shape[2],shape[3],shape[1])).transpose((0,3,1,2))


from nolearn.lasagne.base import TrainSplit


from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold

def _sldict(arr, sl):
    if isinstance(arr, dict):
        return {k: v[sl] for k, v in arr.items()}
    else:
        return arr[sl]
class ownTrainSplit(TrainSplit):

    def __call__(self, X, y, net):
        if self.eval_size:
            if net.regression or not self.stratify:
                kf = KFold(y.shape[0], round(1. / self.eval_size))
            else:
                kf = StratifiedKFold(y, round(1. / self.eval_size))

            train_indices, valid_indices = next(iter(kf))
            X_train, y_train = _sldict(X, train_indices), y[train_indices]
            X_valid, y_valid = _sldict(X, valid_indices), y[valid_indices]
        else:
            X_train, y_train = X, y
            X_valid, y_valid = _sldict(X, slice(len(y), None)), y[len(y):]

        return X_train, X_valid, y_train, y_valid

# MODULES necessary only for the last routine:
import theano
from lasagne.layers import InputLayer
from lasagne.layers import get_output


class segmNeuralNet(modNeuralNet):
    '''
    Modified version of NeuralNet (nolearn), adapted for a segmentation problems.

    Changes:
     - for training: store pixel acc. history
     - for validation: store pixel accuracy and meanIU history

    '''
    def __init__(self,*args,**kwargs):
        eval_size = kwargs.pop('eval_size', 0.1)
        kwargs['train_split'] = ownTrainSplit(eval_size=eval_size,stratify=False)
        kwargs['objective_loss_function'] = categorical_crossentropy_segm
        kwargs['scores_train'] = [('trn pixelAcc', pixel_accuracy)]
        if eval_size!=0:
            kwargs['scores_valid'] = [('val pixelAcc', pixel_accuracy)]
        # kwargs['scores_valid'] = [('val pixelAcc', pixel_accuracy), ('val meanIU', meanIU)]
        super(segmNeuralNet, self).__init__(*args, **kwargs)


    # THIS IS REALLY BAD, but I don't see any solution at the moment...
    def _create_iter_funcs(self, layers, objective, update, output_type):
        y_batch = output_type('y_batch')

        output_layer = layers[-1]
        objective_kw = self._get_params_for('objective')

        loss_train = objective(
            layers, target=y_batch, **objective_kw)
        loss_eval = objective(
            layers, target=y_batch, deterministic=True, **objective_kw)
        predict_proba = get_output(output_layer, None, deterministic=True)
        if not self.regression:
            # THE ONLY CHANGED LINE:
            # predict = predict_proba.argmax(axis=1)
            # accuracy = T.mean(T.eq(predict, y_batch))
            accuracy = loss_eval
        else:
            accuracy = loss_eval

        scores_train = [
            s[1](predict_proba, y_batch) for s in self.scores_train]
        scores_valid = [
            s[1](predict_proba, y_batch) for s in self.scores_valid]

        all_params = self.get_all_params(trainable=True)
        grads = theano.grad(loss_train, all_params)
        for idx, param in enumerate(all_params):
            grad_scale = getattr(param.tag, 'grad_scale', 1)
            if grad_scale != 1:
                grads[idx] *= grad_scale
        update_params = self._get_params_for('update')
        updates = update(grads, all_params, **update_params)

        input_layers = [layer for layer in layers.values()
                        if isinstance(layer, InputLayer)]

        X_inputs = [theano.In(input_layer.input_var, name=input_layer.name)
                    for input_layer in input_layers]
        inputs = X_inputs + [theano.In(y_batch, name="y")]

        train_iter = theano.function(
            inputs=inputs,
            outputs=[loss_train] + scores_train,
            updates=updates,
            allow_input_downcast=True,
            )
        eval_iter = theano.function(
            inputs=inputs,
            outputs=[loss_eval, accuracy] + scores_valid,
            allow_input_downcast=True,
            )
        predict_iter = theano.function(
            inputs=X_inputs,
            outputs=predict_proba,
            allow_input_downcast=True,
            )

        return train_iter, eval_iter, predict_iter


