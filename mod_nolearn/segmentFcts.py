import theano.tensor as T
import numpy as np
import time

# IT SHOULD (or need to) BE COMPLETELY TRANSLATED INTO THEANO LANGUAGE...
# (Not numpy arrays, but Theano tensors)

from sklearn.metrics import confusion_matrix


def compute_mean_IoU_logRegr(predictions, targets):
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
        # print "Done! (%g sec)" %(time.time()-tick)


# # from theano import shared
# def meanIU(prediction, GrTruth):
#     '''
#     Inputs:
#       - prediction: shape (N, class, dimX, dimY) of float32
#       - ground truth: shape (N, dimX, dimY) of int32

#     Return averaged Intersection over union for each sample:
#       - array (N)

#     NOT WORKING, NEEDS THEANO IMPLEMENTATION
#     '''
#     N, C = prediction.shape[:2]
#     predLabels = T.argmax(prediction, axis=1)

#     n_theano = T.itensor3()
#     n = T.ones_like(n_theano)
#     # n = shared(0)
#     # n = np.ones((N,C,C))
#     for cl in range(C):
#         # BAD OPTIMIZED, need better implementation ---------------
#         for i in range(N):
#             idxs = (T.eq(GrTruth[i],cl)).nonzero()
#             predClass, numPixels = T.unique(predLabels[i,idxs],return_counts=True)
#             n[i,cl,predClass] = numPixels

#     diag_n = T.diagonal(n,axis1=1,axis2=2)
#     meanIU = 1./C * T.sum(diag_n / (n.sum(axis=2)+n.sum(axis=1)-diag_n), axis=1 )
#     return meanIU

def pixel_accuracy(prediction_proba, GrTruth):
    '''
    USED FOR A CLASSIFICATION PROBLEM (softmax pixel by pixel)

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
    predictions: ints of shape [N,x,y]

    '''
    right_pixels = np.sum(prediction==targets, axis=(1,2))
    n_pixels = float(targets.shape[1]*targets.shape[2])
    return np.mean(right_pixels/n_pixels)


# def pixel_accuracy_sigmoid(prediction, targets):
#     '''
#     USED FOR A LOGISTIC REGRESSION PROBLEM (sigmoid pixel by pixel)

#     Inputs:
#       - prediction: shape (N, dimX, dimY) of float32. Should come from a sigmoid
#       - ground truth: shape (N, dimX, dimY) of float32 representing GroundTruth or residuals in [0.,1.]

#     Return pixel accuracy [sum(right_pixels)/all_pixels] for each sample:
#       - array (N)

#     '''
#     right_pixels = T.sum( T.lt(T.abs_(prediction-targets), 0.5), axis=(1,2))
#     n_pixels = T.cast(targets.shape[1]*targets.shape[2], 'float32')
#     return T.mean(right_pixels/n_pixels)


# def mean_accuracy(prediction, GrTruth):
#     ...
