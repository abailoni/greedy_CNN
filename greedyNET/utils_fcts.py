import numpy as np

def numpy_check(data):
    return True if isinstance(data, np.ndarray) else False

def join_dict(dict1, dict2):
    new_dict = dict1.copy()
    new_dict.update(dict2)
    return new_dict
