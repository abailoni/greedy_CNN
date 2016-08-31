import cPickle as pickle
import numpy as np
import os
import shutil

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


def copyDirectory(src, dest):
    try:
        if not os.path.exists(dest):
            shutil.copytree(src, dest)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)
