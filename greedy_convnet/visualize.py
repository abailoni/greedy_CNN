import numpy as np
import os

from mod_nolearn.visualize import get_model_data
from greedy_convnet.greedy_net import greedyNet

class get_greedyModel_data(object):

    def __call__(self, inputs,  quantity, **kwargs):
        '''
        INPUTS:
            - input should contain a model or a path (e.g. 'logs/models343/')
            - quantity: e.g. "Training loss sublog"
            - 'mod' should take the following values:
                * subNet name
                * join_layers
        '''
        import six
        if isinstance(inputs, greedyNet):
            self.model = inputs
            self.path = self.model.BASE_PATH_LOG_MODEL
        elif isinstance(inputs, six.string_types):
            self.path = inputs
        else:
            raise ValueError("The input should contain a greedy model or a path")

        # Get list of subNets:
        self.subNet_list = [dir
           for dirpath, dirnames, files in os.walk(self.path)
           for dir in dirnames]

        mod = kwargs.setdefault('mod', 'none')

        if mod!='join_layers' and mod not in self.subNet_list:
            raise ValueError("mod='%s' passed, but it should take the following values: subNet name; 'join_layers'. \nAvailable subNets: ..."%mod)

        out_data = None
        if mod in self.subNet_list:
            out_data, out_plot_kwargs = get_model_data(self.path+mod+'/', quantity)
            # print out_data
            # print quantity
            # raise ValueError("Stop here")
        elif mod=='join_layers':
            layers = [dir for dir in self.subNet_list if dir.startswith('cnv')]
            print layers
            out_plot_kwargs = get_model_data(self.path+layers[0]+'/', quantity)[1]
            outs = [get_model_data(self.path+dir+'/', quantity)[0][1] for dir in layers]
            # outs = np.array(outs, dtype=object)
            all_data = np.concatenate(tuple(outs))
            out_data = [np.arange(all_data.shape[0]), all_data]

        return out_data, out_plot_kwargs


    def compare_boostedNodes(self):
        pass

    def plot_all(self):
        pass
