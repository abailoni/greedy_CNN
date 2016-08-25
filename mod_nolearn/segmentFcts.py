import theano.tensor as T


# IT SHOULD (or need to) BE COMPLETELY TRANSLATED INTO THEANO LANGUAGE...
# (Not numpy arrays, but Theano tensors)


# from theano import shared
def meanIU(prediction, GrTruth):
    '''
    Inputs:
      - prediction: shape (N, class, dimX, dimY) of float32
      - ground truth: shape (N, dimX, dimY) of int32

    Return averaged Intersection over union for each sample:
      - array (N)

    NOT WORKING, NEEDS THEANO IMPLEMENTATION
    '''
    N, C = prediction.shape[:2]
    predLabels = T.argmax(prediction, axis=1)

    n_theano = T.itensor3()
    n = T.ones_like(n_theano)
    # n = shared(0)
    # n = np.ones((N,C,C))
    for cl in range(C):
        # BAD OPTIMIZED, need better implementation ---------------
        for i in range(N):
            idxs = (T.eq(GrTruth[i],cl)).nonzero()
            predClass, numPixels = T.unique(predLabels[i,idxs],return_counts=True)
            n[i,cl,predClass] = numPixels

    diag_n = T.diagonal(n,axis1=1,axis2=2)
    meanIU = 1./C * T.sum(diag_n / (n.sum(axis=2)+n.sum(axis=1)-diag_n), axis=1 )
    return meanIU

def pixel_accuracy(prediction, GrTruth):
    '''
    USED FOR A CLASSIFICATION PROBLEM (softmax pixel by pixel)

    Inputs:
      - prediction: shape (N, class, dimX, dimY) of float32. Should come from a sigmoid or softmax
      - ground truth: shape (N, dimX, dimY) of int. Classes should start from 0.
            Even a tensor (N, class, dimX, dimY) can be accepted.

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

def pixel_accuracy_sigmoid(prediction, targets):
    '''
    USED FOR A LOGISTIC REGRESSION PROBLEM (sigmoid pixel by pixel)

    Inputs:
      - prediction: shape (N, dimX, dimY) of float32. Should come from a sigmoid
      - ground truth: shape (N, dimX, dimY) of float32 representing GroundTruth or residuals in [0.,1.]

    Return pixel accuracy [sum(right_pixels)/all_pixels] for each sample:
      - array (N)

    '''
    right_pixels = T.sum( T.lt(T.abs_(prediction-targets), 0.5), axis=(1,2,3))
    n_pixels = T.cast(targets.shape[1]*targets.shape[2], 'float32')
    return right_pixels/n_pixels



# def mean_accuracy(prediction, GrTruth):
#     ...
