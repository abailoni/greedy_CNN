from mod_nolearn import TrainSplit

from mod_nolearn import NeuralNet
from mod_nolearn.segm.segm_utils import categorical_crossentropy_segm







class segmNeuralNet(NeuralNet):
    '''
    Modified version of NeuralNet (nolearn), adapted for a segmentation problems.

    Changes:
     - applies categorical_crossentropy_segm by default
     - deactivates the sklearn option for representative train/validation data (stratify=False)

     Options: (check default values)
     - train_split
     - objective_loss_function
    '''
    def __init__(self,*args,**kwargs):
        eval_size = kwargs.pop('eval_size', 0.1)
        kwargs.setdefault(
            'train_split',
            TrainSplit(eval_size=eval_size,stratify=False)
        )
        kwargs.setdefault('objective_loss_function', categorical_crossentropy_segm)

        super(segmNeuralNet, self).__init__(*args, **kwargs)
