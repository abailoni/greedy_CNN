import numpy as np
import theano.tensor as T


# IT SHOULD (or need to) BE COMPLETELY TRANSLATED INTO THEANO LANGUAGE...
# (Not numpy arrays, but Theano tensors)


def meanIU(prediction, GrTruth):
    '''
    Inputs:
      - prediction: shape (N, class, dimX, dimY) of float32
      - ground truth: shape (N, dimX, dimY) of int32
    Return averaged Intersection over union for each sample:
      - array (N)

    NOT WORKING, NEEDS THEANO IMPLEMENTATION
    '''
    N, C = prediction.shape[:2] # Are not numbers, can not be used to create a numpy array
    predLabels = np.argmax(prediction, axis=1)

    n = np.ones((N,C,C))
    for cl in range(C):
        # BAD OPTIMIZED, need better implementation ---------------
        for i in range(N):
            idxs = (GrTruth[i]==cl).nonzero()
            predClass, numPixels = np.unique(predLabels[i,idxs],return_counts=True)
            n[i,cl,predClass] = numPixels

    diag_n = np.diagonal(n,axis1=1,axis2=2)
    meanIU = 1./C * np.sum(diag_n / (n.sum(axis=2)+n.sum(axis=1)-diag_n), axis=1 )
    return meanIU

def pixel_accuracy(prediction, GrTruth):
    '''

    Inputs:
      - prediction: shape (N, class, dimX, dimY) of float32. Should come from a sigmoid or softmax
      - ground truth: shape (N, dimX, dimY) of int. Classes should start from 0.
            Actually even a tensor (N, class, dimX, dimY) can be accepted.

    Return pixel accuracy [sum(right_pixels)/all_pixels] for each sample:
      - array (N)

    '''
    predLabels = T.argmax(prediction, axis=1)

    # Check if I have [0,0,1,0,0] instead of a label in GrTruth:
    if prediction.ndim==GrTruth.ndim:
        GrTruth = T.argmax(GrTruth, axis=1)

    right_pixels = T.sum( T.eq(predLabels, GrTruth), axis=(1,2)) # Sum over image dims.
    n_pixels = T.cast(GrTruth.shape[1]*GrTruth.shape[2], 'float32')
    return right_pixels/n_pixels

# def mean_accuracy(prediction, GrTruth):
#     ...

# def segmLoss():
