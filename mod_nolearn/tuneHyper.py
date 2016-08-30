import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

class tune_hyperparams(object):
    def __init__(self, hyperparameters, **kwargs):
        '''
        FOR THE MOMENT IT MAKES SENSE WITH TWO hyperparameters at time.

        Inputs:
          - trainable_object: necessary??
          - hyperparameters: a tuple of lists in this form
                (
                    ['par_name', start, stop, 'linear', np.int8],
                    ['par2_name', ...]
                )

        Options:
          - name
          - num_iterations
          - path_outputs
          - log_filename

        Further possible additions:
          - cancel training of some specific parameters
        '''
        # self.model = trainable_object
        self.hyperparams = hyperparameters
        self.param_names = [hyperparam[0] for hyperparam in hyperparameters]
        self.num_iterations = kwargs.pop('num_iterations', 10)
        self.path_out = kwargs.pop('path_outputs', './')
        self.name = kwargs.pop('name', 'tuningHyper')
        self.log_filename = kwargs.pop('log_filename', self.path_out+self.name+'.txt')


    def __call__(self):
        '''
        Every time the istance is called, a tuning session starts.

        The final results is a dictionary of the results, each containing an array of obtained data.
        '''
        values = self._set_hyperparameters()
        results = []
        for iter_values in values:
            val_dict = {name: value for name,value in zip(self.param_names,iter_values)}
            results.append(self.fit_model(val_dict))

        results = { key: np.array([results[i][key] for i in range(self.num_iterations)]) for key in results[0]}
        self.plot_comparison(values,results,quantity='val_loss')
        self.plot_comparison(values,results,quantity='val_acc')
        self.savefile(values, results)
        self.on_tuning_finished(values, results)

    def fit_model(self, param_values):
        '''
        This function should be replaced by a subclass. It fits the model with the
        given values of hyperparameters and return the results.

        Inputs:
          - param_values: dictionary with the selected hyperparameter values.

        Outputs:
          - dictionary with results. It should contain the following key-words:
               * 'val_loss' (the best)
               * 'val_acc' (the best)
               * other stuff
        '''
        return None

    def on_tuning_finished(self, values, results):
        '''
        After the tuning, for possible further plots or similar.

        '''
        pass

    def _set_hyperparameters(self):
        '''
        Return a numpy array of shape (num_iterations, num_hyperparams)
        '''
        values = []
        for i, hyperparam in enumerate(self.hyperparams):
            start, stop, mode, dtype = hyperparam[1:]
            if mode=='log':
                values.append(10**np.random.uniform(np.log10(start), np.log10(stop), size=self.num_iterations).astype(dtype))
            if mode=='linear':
                values.append(np.random.uniform(start, stop, size=self.num_iterations).astype(dtype))
        return np.array(values).T

    def savefile(self, values, results):
        legend = [['#']+self.param_names+[quantity for quantity in results]]
        header = tabulate(legend, tablefmt='plain')
        results = np.array([results[key] for key in results])
        np.savetxt(self.log_filename, np.column_stack((values,results.T)), header=header)

    def plot_comparison(self, values, results, quantity):
        input_x = values[:,0]
        input_y = values[:,1]
        output = results[quantity]

        # plot
        marker_size = 100
        fig = plt.figure()
        ax = fig.add_subplot(111)
        s = ax.scatter(input_x, input_y, marker_size, c=output, cmap='Greys')
        fig.colorbar(s)

        min_x, max_x = self.hyperparams[0][1:3]
        min_y, max_y = self.hyperparams[1][1:3]
        # range_x = max_x-min_x
        # range_y = max_y-min_y
        # ax.set_xlim([min_x-range_x*0.1,max_x+range_x*0.1])
        # ax.set_ylim([min_y-range_y*0.1,max_y+range_y*0.1])
        ax.set_xlim([min_x,max_x])
        ax.set_ylim([min_y,max_y])
        ax.grid(True)
        ax.set_xlabel(self.param_names[0])
        if self.hyperparams[0][3]=='log':
            ax.set_xscale('log')
        if self.hyperparams[1][3]=='log':
            ax.set_yscale('log')
        ax.set_ylabel(self.param_names[1])
        ax.set_title(quantity)
        fig.set_tight_layout(True)
        fig.savefig(self.path_out+self.name+'_'+quantity+'.pdf')


