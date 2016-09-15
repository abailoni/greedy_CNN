import theano.tensor as T
import numpy as np

from sklearn.metrics import confusion_matrix

from theano.tensor.nnet import categorical_crossentropy
from theano.tensor.nnet import softmax
from various.utils import float32

# ------------------------------------------------
# Adapted non-linearities and objective-losses:
# ------------------------------------------------

# Here the final output is the loss:
def categorical_crossentropy_segm(prediction_proba, targets):
    '''
    MODIFICATIONS:
        - reshape from image-size to array and back
    '''
    shape = T.shape(prediction_proba)
    pred_mod1 = T.transpose(prediction_proba, (0,2,3,1))
    pred_mod = T.reshape(pred_mod1, (-1,shape[1]))
    if prediction_proba.ndim == targets.ndim:
        targ_mod1 = T.transpose(targets,(0,2,3,1))
        targ_mod = T.reshape(targ_mod1,(-1,shape[1]))
    else:
        targ_mod = T.reshape(targets, (-1,))
    results = categorical_crossentropy(pred_mod, targ_mod)


    results = T.reshape(results, (shape[0],shape[2],shape[3]))



    # QUICK IMPLEMENTATION FOR TWO SPECIFIC CLASSES. NEEDS GENERALIZATION
    # Weights depending on class occurency:
    weights = (1.02275, 44.9647)
    cars_indx, not_cars_indx = T.nonzero(targets), T.nonzero(T.eq(targets,0))
    T.set_subtensor(results[cars_indx], results[cars_indx]*float32(weights[1]) )
    T.set_subtensor(results[not_cars_indx], results[not_cars_indx]*float32(weights[0]) )


    return T.sum(results, axis=(1,2))



# This is the non-linearity, giving the probabilities:
def softmax_segm(x):
    '''
    MODIFICATIONS:
        - reshape from image-size to array and back
    '''
    shape = T.shape(x)
    x_mod = T.transpose(x, (0,2,3,1))
    x_mod = T.reshape(x_mod, (-1,shape[1]))
    results = softmax(x_mod)
    results = T.reshape(results, (shape[0],shape[2],shape[3],shape[1]))
    return T.transpose(results, (0,3,1,2))





# ------------------------------------------------
# Segmentation-related quantities:
# ------------------------------------------------

def compute_mean_IoU_logRegr(predictions, targets):
    '''
    Compute the Intersection over Union.

    Issues:
        - Still performing a loop over the batch samples...
        - Theano implementation missing
    '''
    predictions = predictions.squeeze()
    N = predictions.shape[0]
    predictions = predictions.reshape((N,-1)).astype(np.int8)
    targets = targets.reshape((N,-1)).astype(np.int8)
    IoU = np.empty(N)
    for i in range(N):
        cnf_mat = confusion_matrix(targets[i], predictions[i], labels=[0,1]).astype(np.float32)
        diag = np.diagonal(cnf_mat)
        IoU[i] = 1./2. * np.sum(diag / (cnf_mat.sum(axis=1)+cnf_mat.sum(axis=0)-diag) )
    return np.mean(IoU)


class mean_IoU(object):
    '''
    Helper class for computing IoU
    '''
    def __init__(self, X_train, X_valid, y_train, y_valid, **kwargs):
        self.epochs = 0
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.out = []
        self.n_cl = 2

    def __call__(self, net, train_history_):
        # tick = time.time()
        # print "Computing IoU..."
        self.epochs += 1
        X_tr = self.X_train
        X_val = self.X_valid
        pred_train = net.predict(X_tr)
        pred_valid = net.predict(X_val)
        IoU_tr = compute_mean_IoU_logRegr(pred_train, self.y_train)
        IoU_val = compute_mean_IoU_logRegr(pred_valid, self.y_valid)
        net.train_history_[-1]['Train IoU'] = IoU_tr
        net.train_history_[-1]['Valid IoU'] = IoU_val


def pixel_accuracy(prediction_proba, GrTruth):
    '''
    THEANO IMPLEMENTATION OF PIXEL ACCURACY
    (can be used as training score in nolearn module)

    Inputs:
      - prediction_proba: shape (N, class, dimX, dimY) of float32. Should come from a sigmoid or softmax
      - ground truth: shape (N, dimX, dimY) of int. Classes should start from 0.
            Even a tensor (N, class, dimX, dimY) can be accepted.

    Return pixel accuracy [sum(right_pixels)/all_pixels] for each sample:
      - array (N)

    '''
    predLabels = T.argmax(prediction_proba, axis=1)

    # Check if I have [0,0,1,0,0] instead of a label in GrTruth:
    if prediction_proba.ndim==GrTruth.ndim:
        GrTruth = T.argmax(GrTruth, axis=1)

    right_pixels = T.sum( T.eq(predLabels, GrTruth), axis=(1,2)) # Sum over image dims.
    n_pixels = T.cast(GrTruth.shape[1]*GrTruth.shape[2], 'float32')
    return T.mean(right_pixels/n_pixels)

def pixel_accuracy_np(prediction, targets):
    '''
    Similar to pixel_accuracy(), but implemented in numpy.

    Inputs:
        - predictions: ints of shape (N,dim_x,dim_y) or float of shape (N,N_classes,dim_x,dim_y)
        - GrTruth (N,dim_x,dim_y)

    '''
    if len(prediction.shape)==4:
        prediction = np.argmax(prediction, axis=1)
    right_pixels = np.sum(prediction==targets, axis=(1,2))
    n_pixels = float(targets.shape[1]*targets.shape[2])
    return np.mean(right_pixels/n_pixels)

