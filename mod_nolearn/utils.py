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


def deleteDirectory(path):
    '''
    Delete directory if there.
    '''
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not deleted. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not deleted. Error: %s' % e)


def check_decay(start, N_epochs, decay_rate, mod='log'):
    results = np.empty(N_epochs+1)
    results[0] = start
    for i in range(N_epochs):
        old_value = results[i]
        if mod=='lin':
            results[i+1] = old_value/(1.+decay_rate*(i+1))
        elif mod=='log':
            results[i+1] = old_value*np.exp(-decay_rate*(i+1))
    return results
