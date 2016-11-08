import numpy as np
from mod_nolearn import BatchIterator
from various.bin_ndarray import bin_ndarray
# import time

class BatchIterator_Greedy(BatchIterator):
    '''
    It resizes the GT accordingly

    Inputs: (in addition to usual ones of BatchIterator)
        - 'layer_output_shape'
    '''

    def __init__(self, layer_output_shape, *args, **kwargs):
        self.layer_output_shape = layer_output_shape[-2:]
        super(BatchIterator_Greedy, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):
        # Scale down the ground truth:
        #

        # import numpy as np
        # yb = np.arange(0,256,1).reshape((16,16))
        # yb = np.tile(yb, (20,1,1))
        # print yb[0]
        # self.layer_output_shape = (4,4)
        # # self.layer_output_shape = [i/2.  for i in self.layer_output_shape]


        #
        #
        if yb is not None:
            GT_spatial_outputShape = yb.shape[-2:]
            if GT_spatial_outputShape!=self.layer_output_shape:
                temp = bin_ndarray(yb, (yb.shape[0], self.layer_output_shape[0], self.layer_output_shape[1]), operation='avg')
                # Round to 0 or 1:
                yb = np.around(temp).astype(np.int32)

        return Xb, yb


class BatchIterator_Greedy_old(BatchIterator):
    '''
    It replaces the batch inputs Xb with the output of the previous computed layers.

    Inputs: (in addition to usual ones of BatchIterator)
        - 'previous_layers': instance of nolearn NeuralNet
    '''

    def __init__(self, *args, **kwargs):
        self.previous_layers = kwargs.pop('previous_layers', None)
        super(BatchIterator_Greedy, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):
        # Process inputs:
        if self.previous_layers:
            # tick = time.time()
            Xb = self.previous_layers.predict_proba(Xb)
            # print "Tock: %g s" %(time.time()-tick)

        return Xb, yb
