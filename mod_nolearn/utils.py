import cPickle as pickle

def pickle_model(net, filename):
    with open(filename, 'wb') as f:
        pickle.dump(net, f, -1)

def restore_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
