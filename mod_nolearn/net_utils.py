import numpy as np

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
