import lasagne
import theano
from theano import tensor as T

data_size = (None, 3, 32, 32)   # Batch size x Img Channels x Height x Width
output_size = 10

prediction = T.tensor3('prediction')
targets = T.tensor3('targets')

right_pixels = T.sum( T.lt(T.abs_(prediction-targets), 0.5), axis=(1,2))
n_pixels = T.cast(targets.shape[1]*targets.shape[2], 'float32')
output T.mean(right_pixels/n_pixels)

train_fn = theano.function([prediction, targets], output, name='train')

