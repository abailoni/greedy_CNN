import cPickle as pickle
import numpy as np
import os

def pickle_model(net, filename):
    with open(filename, 'wb') as f:
        pickle.dump(net, f, -1)

def restore_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def float32(k):
    return np.cast['float32'](k)

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
