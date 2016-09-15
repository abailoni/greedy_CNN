from mod_nolearn import BatchIterator
# import time

class BatchIterator_Greedy(BatchIterator):
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

