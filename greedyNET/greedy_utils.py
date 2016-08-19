import time
import numpy as np

from scipy.fftpack import dct

from nolearn.lasagne import BatchIterator

class BatchIterator_Greedy(BatchIterator):
    '''
    It modifies the inputs using the processInput() class.

    Inputs:
      - processInput (None): to apply DCT or previous fixed layers. Example of input: processInput(DCT_size=4)
      - other usual options: batch_size, shuffle
    '''

    def __init__(self, *args, **kwargs):
        self.processInput = kwargs.pop('processInput', None)
        super(BatchIterator_Greedy, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):
        # Process inputs:
        if self.processInput:
            Xb = self.processInput(Xb)

        return Xb, yb


### INPUT FUNCTION:

class processInput(object):
    '''
    Compute the input. The function is called each time we choose a batch of data.

    It should have an attribute with the dimensions of the output! (not much the spatial ones, but the number of channels..)

    Arguments:
     - DCT_size (7): size of the DCT filter, the output will have 7*7*3 filters. The input can be set to None and then nothing will be done.
     - fixed_layers (None): optional network representing the previous learned and fixed layers (the DCT filters should be included...?)
    '''
    def __init__(self, **kwargs):
        self.DCT_size = kwargs.pop('DCT_size', 7)
        self.image_channels = kwargs.pop('image_channels', 3)
        self.fixed_layers = kwargs.pop('fixed_layers', None)
        pr_layrs = self.fixed_layers
        if kwargs:
            raise Warning('Additional not necesary arguments to handleInput have been passed')
        # Detect number of outputs if you have previous layers:
        # if pr_layrs:
        #     if isinstance(pr_layrs, Boost_LogRegr):
        #         pr_layrs.net.layers_[-1][]

        # TEMP:
        self.output_channels = self.image_channels


    def __call__(self, batch_input):
        if self.fixed_layers!=None:
            batch_output = self.fixed_layers.predict_proba(batch_input)
        elif self.DCT_size:
            batch_output = self.apply_DCT(batch_input)
        else:
            batch_output = batch_input

        return batch_output

    def apply_DCT(self, batch_input):
        '''
        Size of the batch_input: (N,3,dim_x,dim_y)
        It needs to be implemented at least in cython... Or..?
        '''
        N, channels, dim_x, dim_y = batch_input.shape
        pad = self.DCT_size/2

        temp = np.empty((N,channels*self.DCT_size,dim_x,dim_y+2*pad)) #dct along one dim.
        output = np.empty((N,channels*self.DCT_size**2,dim_x,dim_y), dtype=np.float32)
        padded_input = np.pad(batch_input,pad_width=((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')

        tick = time.time()
        for i in range(dim_x):
            # Note that (i,j) are the center of the filter in input, but are the top-left corner coord. in the padded_input
            if i%10==0:
                print i
            temp[:,:,i,:] = np.reshape(dct(padded_input[:,:,i:,:], axis=2, n=self.DCT_size), (N,-1,dim_y+2*pad))
        for j in range(dim_y):
            if j%10==0:
                print j
            output[:,:,:,j] = dct(temp[:,:,:,j:], axis=3, n=self.DCT_size).reshape((N,-1,dim_x)).astype(np.float32)
        tock = time.time()
        print "Conversion: %g sec." %(tock-tick)
        return output
