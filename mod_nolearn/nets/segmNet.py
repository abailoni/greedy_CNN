

from nolearn.lasagne import NeuralNet
from ..segmentFcts import meanIU, pixel_accuracy

class segmNeuralNet(NeuralNet):
    '''
    Modified version of NeuralNet (nolearn), adapted for a segmentation problems.

    Changes:
     - for training: store pixel accuracy history
     - for validation: store pixel accuracy and meanIU hostory
    '''
    def __init__(self,*args,**kwargs):
        kwargs['scores_train'] = [('pixAcc', pixel_accuracy)]
        kwargs['scores_valid'] = [('pixAcc', pixel_accuracy), ('meanIU', meanIU)]
        super(NeuralNet, self).__init__(*args, **kwargs)

