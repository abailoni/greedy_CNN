import cPickle as pickle
import numpy as np
import os
import shutil
import os.path


def pickle_model(net, filename):
    with open(filename, 'wb') as f:
        pickle.dump(net, f, -1)

def restore_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def numpy_check(data):
    return True if isinstance(data, np.ndarray) else False


def join_dict(dict1, dict2):
    new_dict = dict1.copy()
    new_dict.update(dict2)
    return new_dict



def float32(k):
    return np.cast['float32'](k)

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def check_file(path_file):
    return os.path.isfile(path_file)


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



