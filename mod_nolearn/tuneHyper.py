import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import json
import os


def scatter_plot(folder, tun_ID=None, quantity=None, exclude=None):
    '''
    Take the log file given as input from tune_hyperparams() procedure and print scatter plots.

    Inputs:
        - folder with tuning files
        - if any ID is passed, all tuning data are merged (not implemented)
        - name of the quantity (if None, all are printed)
        - exclude N big values from the plot

    Remark: if more files are used and merge, the same quantities should be present in each file.
    '''
    # Collect data:
    info_files = []
    data = []
    if tun_ID:
        data.append(np.loadtxt(folder+'/log-%d.txt' %tun_ID))
        with open(folder+'/info-tuning_%d.txt' %tun_ID) as info_file:
            info_files.append(json.load(info_file))
    else:
        # IMPLEMENT REGEX!
        log_list = [os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(folder)
            for f in files if f.startswith('log-')]
        info_list = [os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(folder)
            for f in files if f.startswith('info-tuning_')]
        for info_name in info_list:
            with open(info_name) as info_file:
                info_files.append(json.load(info_file))
        data = [np.loadtxt(filename) for filename in log_list]
    outputs = info_files[0]['outputs']

    # Merge data:
    x_min = min([info_files[i]['hyperparams'][0][1] for i in range(len(info_files))])
    x_max = max([info_files[i]['hyperparams'][0][2] for i in range(len(info_files))])
    y_min = min([info_files[i]['hyperparams'][1][1] for i in range(len(info_files))])
    y_max = max([info_files[i]['hyperparams'][1][2] for i in range(len(info_files))])
    x_info = info_files[0]['hyperparams'][0]
    y_info = info_files[0]['hyperparams'][1]
    all_data = np.row_stack((data[i] for i in range(len(data))))

    # Delele NaN:
    nan_indx = (np.isnan(all_data[:,3])).nonzero()
    data_indx = np.logical_not(np.isnan(all_data[:,3])).nonzero()
    nan_data = np.delete(all_data,data_indx,axis=0)
    nan_input_x = nan_data[:,1]
    nan_input_y = nan_data[:,2]


    all_data = np.delete(all_data,nan_indx,axis=0)


    # Order data:
    all_data = all_data[all_data[:,3].argsort()]
    if exclude:
        all_data = all_data[:-exclude]
    input_x = all_data[:,1]
    input_y = all_data[:,2]

    # Collect results:
    if quantity:
        quantities = [quantity]
        column = 3 + outputs.index(quantity)
        results = [all_data[:,column]]
    else:
        results = all_data[:,3:].T
        quantities = outputs

    for quantity, result in zip(quantities, results):
        # plot
        marker_size = 100
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(input_x[:1], input_y[:1], s=marker_size+100, c=result[:1], cmap='autumn', alpha=0.8, edgecolor='r', marker=(4, 0))
        s = ax.scatter(input_x, input_y, s=marker_size, c=result, cmap='jet') #Greys
        ax.scatter(nan_input_x, nan_input_y, marker_size, marker='x')

        fig.colorbar(s)

        ax.set_xlim([x_min,x_max])
        ax.set_ylim([y_min,y_max])
        ax.grid(True)
        ax.set_xlabel(x_info[0])
        ax.set_ylabel(y_info[0])
        if x_info[3]=='log':
            ax.set_xscale('log')
        if y_info[3]=='log':
            ax.set_yscale('log')
        string = "  -  Best: %d, %.5f, %.5f" %(all_data[0,0], input_x[0], input_y[0])
        ax.set_title(quantity+string)
        fig.set_tight_layout(True)
        name = quantity+'-'+tun_ID+'.pdf' if tun_ID else quantity+'-ALL'
        if exclude:
            name+='-%d' %exclude
        fig.savefig(folder+'/'+name+'.pdf')

def compare_stuff(folder, collect_function, quantity, tun_ID=None, exclude=None, plot_kwargs={}):
    '''
    Take all the models trained in a tuning procedure and plot some quantity (e.g.
        the loss) for every model.
    '''
    # Collect data:
    info_files = []
    data = []
    if tun_ID:
        data.append(np.loadtxt(folder+'/log-%d.txt' %tun_ID))
        with open(folder+'/info-tuning_%d.txt' %tun_ID) as info_file:
            info_files.append(json.load(info_file))
    else:
        # IMPLEMENT REGEX!
        log_list = [os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(folder)
            for f in files if f.startswith('log-')]
        info_list = [os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(folder)
            for f in files if f.startswith('info-tuning_')]
        for info_name in info_list:
            with open(info_name) as info_file:
                info_files.append(json.load(info_file))
        data = [np.loadtxt(filename) for filename in log_list]

    # # Merge data:
    # x_min = min([info_files[i]['hyperparams'][0][1] for i in range(len(info_files))])
    # x_max = max([info_files[i]['hyperparams'][0][2] for i in range(len(info_files))])
    # y_min = min([info_files[i]['hyperparams'][1][1] for i in range(len(info_files))])
    # y_max = max([info_files[i]['hyperparams'][1][2] for i in range(len(info_files))])
    # x_info = info_files[0]['hyperparams'][0]
    # y_info = info_files[0]['hyperparams'][1]
    all_data = np.row_stack((data[i] for i in range(len(data))))

    # Delete NaN:
    # print all_data.shape
    nan_indx = (np.isnan(all_data[:,3])).nonzero()
    all_data = np.delete(all_data,nan_indx,axis=0)
    # print nan_indx
    # print all_data.shape

    # Order data:
    all_data = all_data[all_data[:,3].argsort()]
    if exclude:
        all_data = all_data[:-exclude]
    # input_x = all_data[:,1]
    # input_y = all_data[:,2]
    models = all_data[:,0]

    outputs = [ collect_function(model, quantity, info_files[0]['path_out']) for model in models]
    N = all_data.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    vis.plot_fcts_PRO(ax, [output[0] for output in outputs], [output[1] for output in outputs], labels=models.astype(np.uint32), **plot_kwargs)
    # vis.plot_fcts_show(axis, x, ys,     )
    name = quantity+'-'+tun_ID+'.pdf' if tun_ID else quantity+'-ALL'
    if exclude:
        name+='-%d' %exclude
    fig.savefig(folder+'/'+name+'.pdf')

def analyse_model(folder, collect_function, quantities, model_ID, plot_kwargs={}):
    '''
    Take all the models trained in a tuning procedure and plot some quantity (e.g.
        the loss) for every model.
    '''
    outputs = [ collect_function(model_ID, quantity, folder+'/') for quantity in quantities]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    vis.plot_fcts_PRO(ax, [output[0] for output in outputs], [output[1] for output in outputs], labels=quantities, **plot_kwargs)
    # vis.plot_fcts_show(axis, x, ys,     )
    fig.savefig(folder+'/model_%d/plot_%s.pdf' %(model_ID, quantities[0]))



import mod_nolearn.utils as utils
import mod_nolearn.visualize as vis


class tune_hyperparams(object):
    def __init__(self, hyperparameters, outputs, **kwargs):
        '''
        FOR THE MOMENT IT MAKES SENSE WITH TWO hyperparameters at time.

        Inputs:
          - hyperparameters: a tuple of lists in this form
                (
                    ['par_name', start, stop, 'linear', np.int8],
                    ['par2_name', ...]
                )
          - outputs: keys-name for outputs

        Options:
          - name
          - num_iterations
          - path_outputs
          - log_filename

        Further possible additions:
          - cancel training of some specific parameters
        '''
        self.hyperparams = hyperparameters
        self.outputs = outputs
        self.param_names = [hyperparam[0] for hyperparam in hyperparameters]
        self.num_iterations = kwargs.pop('num_iterations', 10)
        self.name = kwargs.pop('name', 'tuningHyper')
        self.folder_out = kwargs.pop('folder_out', './'+self.name)
        self.path_out = self.folder_out+'/'+self.name+'/'
        utils.create_dir(self.path_out)
        self.tune_id = np.random.randint(1e2,1e3-1)
        self.log_filename = self.path_out+'log-%d.txt' %self.tune_id
        self.plot_flag = kwargs.pop('plot', True)

        # Log info:
        info = {}
        info['name'] = self.name
        info['path_out'] = self.path_out
        info['hyperparams'] = [param[:-1] for param in self.hyperparams]
        info['outputs'] = outputs
        json.dump(info, file(self.path_out+'info-tuning_%d.txt' %self.tune_id, 'w'))

    def __call__(self):
        '''
        Every time the istance is called, a tuning session starts.

        The final results is a dictionary of the results, each containing an array of obtained data.
        '''
        model_ids, values = self._set_hyperparameters()
        results = []
        for k, iter_values in enumerate(values):
            val_dict = {name: value for name,value in zip(self.param_names,iter_values)}
            print "\n\n=========================="
            print "TUNING %d: (%d)" %(model_ids[k], self.tune_id)
            print val_dict
            print "=========================="
            results.append(self.fit_model(val_dict, "model_%d" %(model_ids[k])))
            results_mod = { key: np.array([results[i][key] for i in range(k+1)]) for key in results[0]}
            self.savefile(model_ids, values, results_mod)

        if self.plot_flag:
            raise Warning("Wrong implemented...")
            self.plot_comparison(values,results_mod,quantity='val_loss')
            self.plot_comparison(values,results_mod,quantity='val_acc')
        self.on_tuning_finished(values, results_mod)

    def fit_model(self, param_values, model_name):
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
            if isinstance(start, list):
                # Choose from some passed values:
                array = np.array(start)
                if mode=='shuffle':
                    array = np.random.shuffle(array)
                if self.num_iterations<array.shape[0]:
                    array = array[:self.num_iterations]
                else:
                    import warnings
                    warnings.warn("Not enough values passed for the number of iterations. Doing the passed values, but values repetition still not implemented.")
                    self.num_iterations = array.shape[0]
                values.append(array)
            elif stop:
                if mode=='log':
                    values.append(10**np.random.uniform(np.log10(start), np.log10(stop), size=self.num_iterations).astype(dtype))
                if mode=='linear':
                    values.append(np.random.uniform(start, stop, size=self.num_iterations).astype(dtype))
            else:
                # Just one repeated value, for test:
                values.append(np.array([start]*self.num_iterations))
        model_ids = np.random.randint(1e5,1e6-1,size=self.num_iterations)
        return model_ids, np.array(values).T

    def savefile(self, model_ids, values, results):
        legend = [['ModName']+self.param_names+[quantity for quantity in self.outputs]]
        header = tabulate(legend, tablefmt='plain')
        results = np.array([results[key] for key in self.outputs]).T
        if results.shape[0]<values.shape[0]:
            # Cut values:
            values = values[:results.shape[0],:]
            model_ids = model_ids[:results.shape[0]]
        fmt="%d"+"\t%.5e"*(len(legend[0])-1)
        data = np.column_stack((model_ids,values,results))
        # Pretty bad for the index.... Order wrt valid loss:
        data = data[data[:,len(self.param_names)+2].argsort()]
        np.savetxt(self.log_filename, data, header=header, fmt=fmt)

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


