import matplotlib

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def plot_loss(net, name_file="loss.pdf"):
    fig = plt.figure()
    ax=fig.add_subplot(111)
    train_loss = [row['train_loss'] for row in net.train_history_]
    valid_loss = [row['valid_loss'] for row in net.train_history_]
    ax.plot(train_loss, label='train loss')
    ax.plot(valid_loss, label='valid loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend(loc='best')
    fig.savefig(name_file)

from itertools import product
import numpy as np

def plot_conv_weights_mod(layer, figsize=(6, 6)):
    """Plot the weights of a specific layer.
    Only really makes sense with convolutional layers.
    Parameters
    ----------
    layer : lasagne.layers.Layer
    """
    W = layer.W.get_value()
    shape = W.shape
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows

    figs, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
        if i >= shape[0]:
            break
        axes[r, c].imshow(W[i].transpose((1,2,0), cmap=None), interpolation='nearest')
    return plt


def plot_images(images, figsize=(6, 6)):
    shape = images.shape
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows

    # # Readjust mean image and variance:
    # mean_train = np.reshape([[[ 73.15835921 , 82.90891754, 72.39239876]]],(3,1,1)) /255.
    # var_train = np.reshape([[[ 2237.79144756,  2326.24575092, 2256.68620499]]],(3,1,1)) / 255.**2
    # images = np.add(images, mean_train)
    # images = np.multiply(images, np.sqrt(var_train))

    figs, axes = plt.subplots(nrows, ncols, squeeze=False)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
        if i >= shape[0]:
            break
        axes[r, c].imshow(images[i].transpose((1,2,0)))
    return plt


def plot_GrTruth(images, figsize=(6, 6)):
    shape = images.shape
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows


    figs, axes = plt.subplots(nrows, ncols, squeeze=False)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
        if i >= shape[0]:
            break
        s = axes[r, c].imshow(images[i], cmap='bwr', interpolation='nearest')
        # figs.colorbar(s)
    figs.subplots_adjust(right=0.8)
    cbar_ax = figs.add_axes([0.85, 0.15, 0.05, 0.7])
    figs.colorbar(s, cax=cbar_ax)
    return plt


from lasagne.layers import get_all_param_values

def print_weight_distribution(net, layer_name=None):
    n_layers = net.layers.len()
    layers_names = [net.layers[i][1]['name'] for i in range(1,n_layers)]
    mean, std, weights = {}, {}, {}
    for name in layers_names:
        if "conv" in name:
            layer = net.layers_[name]
            W, _ = get_all_param_values(layer)
            mean[name], std[name], weights[name] = W.mean(), W.std(), W

    if layer_name:
        print "Mean: %g; \tstd: %g" %(mean[layer_name], std[layer_name])
        # Plot?
    else:
        for name in mean:
            print "Layer %s: \tMean: %g; \tstd: %g" %(name, mean[name], std[name])



# ------------------------------------------
# GENERAL things TO PLOT STUFF:
# ------------------------------------------

def check_matplot_arguments(type_of_plot,**kargs):
    if not "labels" in kargs:
        kargs["labels"] = [""]*200
    if not "legend" in kargs:
        kargs["legend"] = "best"
    if not "colors" in kargs:
        kargs["colors"] = None

    # For ellipses:
    if type_of_plot=="ellipses":
        if not "ticks_size" in kargs:
            kargs["ticks_size"] = 9
        if not "label_size" in kargs:
            kargs["label_size"] = kargs["ticks_size"]
        if not "opacities" in kargs:
            kargs["opacities"] = [0.3]*200

    # For line plots (particularly spectra...)
    elif type_of_plot=="linePlot":
        if not "ticks_size" in kargs:
            kargs["ticks_size"] = 13
        if not "label_size" in kargs:
            kargs["label_size"] = kargs["ticks_size"]
        if not "opacities" in kargs:
            kargs["opacities"] = [1.]*200
        if not "log" in kargs:
            kargs["log"] = ""
        if not "lineStyles" in kargs:
            kargs["lineStyles"] = ['-']*200
        if not "xyLabels" in kargs:
            kargs["xyLabels"] = ['$k$ [$h$/Mpc]', '$P(k)$ [(Mpc/$h$)$^3$]']
        if not "xrange" in kargs:
            kargs["xrange"] = 0
        if not "yrange" in kargs:
            kargs["yrange"] = 0
        if not 'grid' in kargs:
            kargs['grid'] = True
    return kargs

def plot_fcts(axis, x, ys, **plot_kargs):
    plot_kargs = check_matplot_arguments("linePlot",**plot_kargs)

    N = len(ys)
    if not plot_kargs['colors']:
        plot_kargs['colors']=cm.rainbow(np.linspace(0,1,N))

    # matplotlib.rcParams['text.usetex'] = True
    font_style = {'weight' : 'normal', 'size': plot_kargs['ticks_size'],'family':'serif','serif':['Palatino']}
    matplotlib.rc('font',**font_style)

    for y, label, color, lineStyle, opacity in zip(ys,plot_kargs['labels'],plot_kargs['colors'],plot_kargs['lineStyles'],plot_kargs['opacities']):
        axis.plot(x,y,lineStyle,color=color,label=r'%s'%(label),alpha=opacity)
    if plot_kargs['grid']==True:
        axis.grid(True)
    axis.legend(loc=plot_kargs['legend'])
    if 'x' in plot_kargs['log']:
        if 'symx' in plot_kargs['log']:
            axis.set_xscale('symlog')
        else:
            axis.set_xscale('log')
    if 'symy' in plot_kargs['log']:
        axis.set_yscale('symlog')
    elif 'symx' in plot_kargs['log']:
        print "boh"
    elif 'y' in plot_kargs['log']:
        axis.set_yscale('log')
    if plot_kargs["xrange"]!=0:
        axis.set_xlim(plot_kargs["xrange"])
    if plot_kargs["yrange"]!=0:
        axis.set_ylim(plot_kargs["yrange"])
    axis.set_xlabel(r'%s' %(plot_kargs['xyLabels'][0]),fontsize=plot_kargs["label_size"])
    axis.set_ylabel(r'%s' %(plot_kargs['xyLabels'][1]),fontsize=plot_kargs["label_size"])

    return axis




def plot_fcts_show(x, ys, **plot_kargs):
    '''
    if not "labels" in kargs:
        kargs["labels"] = [""]*20
    if not "legend" in kargs:
        kargs["legend"] = "best"
    if not "colors" in kargs:
        kargs["colors"] = ['r','b','g','y','m','k']*20

        if not "ticks_size" in kargs:
            kargs["ticks_size"] = 13
        if not "label_size" in kargs:
            kargs["label_size"] = kargs["ticks_size"]
        if not "opacities" in kargs:
            kargs["opacities"] = [1.]*20
        if not "log" in kargs:
            kargs["log"] = ""
        if not "lineStyles" in kargs:
            kargs["lineStyles"] = ['-']*20
        if not "xyLabels" in kargs:
            kargs["xyLabels"] = ['$k$ [$h$/Mpc]', '$P(k)$ [(Mpc/$h$)$^3$]']
        if not "xrange" in kargs:
            kargs["xrange"] = 0
        if not "yrange" in kargs:
            kargs["yrange"] = 0
        if not 'grid' in kargs:
            kargs['grid'] = True

    '''
    axis = plt.subplot(111)

    plot_kargs = check_matplot_arguments("linePlot",**plot_kargs)

    N = len(ys)
    if not plot_kargs['colors']:
        plot_kargs['colors']=cm.rainbow(np.linspace(0,1,N))


    # matplotlib.rcParams['text.usetex'] = True
    font_style = {'weight' : 'normal', 'size': plot_kargs['ticks_size'],'family':'serif','serif':['Palatino']}
    matplotlib.rc('font',**font_style)

    for y, label, color, lineStyle, opacity in zip(ys,plot_kargs['labels'],plot_kargs['colors'],plot_kargs['lineStyles'],plot_kargs['opacities']):
        axis.plot(x,y,lineStyle,color=color,label=r'%s'%(label),alpha=opacity)
    if plot_kargs['grid']==True:
        axis.grid(True)
    axis.legend(loc=plot_kargs['legend'])
    if 'x' in plot_kargs['log']:
        if 'symx' in plot_kargs['log']:
            axis.set_xscale('symlog')
        else:
            axis.set_xscale('log')
    if 'symy' in plot_kargs['log']:
        axis.set_yscale('symlog')
    elif 'symx' in plot_kargs['log']:
        print "boh"
    elif 'y' in plot_kargs['log']:
        axis.set_yscale('log')
    if plot_kargs["xrange"]!=0:
        axis.set_xlim(plot_kargs["xrange"])
    if plot_kargs["yrange"]!=0:
        axis.set_ylim(plot_kargs["yrange"])
    axis.set_xlabel(r'%s' %(plot_kargs['xyLabels'][0]),fontsize=plot_kargs["label_size"])
    axis.set_ylabel(r'%s' %(plot_kargs['xyLabels'][1]),fontsize=plot_kargs["label_size"])

    return plt

def plot_fcts_PRO(axis, xs, ys, **plot_kargs):
    plot_kargs = check_matplot_arguments("linePlot",**plot_kargs)

    N = len(ys)
    if not plot_kargs['colors']:
        plot_kargs['colors']=cm.rainbow(np.linspace(0,1,N))
        # if not isinstance(plot_kargs['colors'], list):
        #     plot_kargs['colors'] = [plot_kargs['colors']]


    # matplotlib.rcParams['text.usetex'] = True
    font_style = {'weight' : 'normal', 'size': plot_kargs['ticks_size'],'family':'serif','serif':['Palatino']}
    matplotlib.rc('font',**font_style)

    for x, y, label, color, lineStyle, opacity in zip(xs,ys,plot_kargs['labels'],plot_kargs['colors'],plot_kargs['lineStyles'],plot_kargs['opacities']):
        axis.plot(x,y,lineStyle,color=color,label=r'%s'%(label),alpha=opacity)
    if plot_kargs['grid']==True:
        axis.grid(True)
    axis.legend(loc=plot_kargs['legend'])
    if 'x' in plot_kargs['log']:
        if 'symx' in plot_kargs['log']:
            axis.set_xscale('symlog')
        else:
            axis.set_xscale('log')
    if 'symy' in plot_kargs['log']:
        axis.set_yscale('symlog')
    elif 'symx' in plot_kargs['log']:
        print "boh"
    elif 'y' in plot_kargs['log']:
        axis.set_yscale('log')
    if plot_kargs["xrange"]!=0:
        axis.set_xlim(plot_kargs["xrange"])
    if plot_kargs["yrange"]!=0:
        axis.set_ylim(plot_kargs["yrange"])
    axis.set_xlabel(r'%s' %(plot_kargs['xyLabels'][0]),fontsize=plot_kargs["label_size"])
    axis.set_ylabel(r'%s' %(plot_kargs['xyLabels'][1]),fontsize=plot_kargs["label_size"])

    return axis


# ###############################################
#   ADVANCED PLOTTING:
# ###############################################

import os

from mod_nolearn.nets.modNeuralNet import modNeuralNet as modNet
from greedyNET.nets.greedyNet import greedyRoutine

def get_model_data(inputs, quantity):
    '''
    Inputs:
        - input should contain a model or a path
    '''
    # Import files:
    import six
    if isinstance(inputs, modNet):
        model = inputs
        path = model.logs_path
        # This should be definitely improved...
        log_file_path = path+'log.txt'
        sublog_file_path = path+'sub_log.txt'
    elif isinstance(inputs, six.string_types):
        path = inputs
        log_file_path = path+'log.txt'
        sublog_file_path = path+'sub_log.txt'
    else:
        raise ValueError("The input should contain a model or a path")

    # --------------------
    # Analyze quantity:
    # --------------------
    # Should be improved with data from info_file...
    out_data, out_plot_kwargs = None, {}
    qnt = quantity.lower()
    if 'sublog' in qnt:
        if 'val' in qnt:
            raise ValueError("Validation data not present in sublogs")
        data = np.loadtxt(sublog_file_path)
        data = data.reshape((data.shape[0],-1))
        iterations = np.arange(data.shape[0])
        quantity = quantity.replace("sublog", "")
        out_plot_kwargs['xyLabels'] = ['Iterations', quantity]
        if 'loss' in qnt:
            out_data = [iterations, data[:,0]]
        elif 'acc' in qnt:
            out_data = [iterations, data[:,1]]
    else:
        data = np.loadtxt(log_file_path)
        if len(data.shape)==1:
            data = data.reshape((1,data.shape[0]))
        epochs = data[:,0]
        # quantity = quantity.replace("log", "")
        out_plot_kwargs['xyLabels'] = ['Epochs', quantity]
        if 'loss' in qnt:
            if 'train' in qnt:
                # print data.shape
                out_data = [epochs, data[:,1]]
            if 'val' in qnt:
                out_data = [epochs, data[:,2]]
        if 'acc' in qnt:
            if 'train' in qnt:
                out_data = [epochs, data[:,5]]
            if 'val' in qnt:
                out_data = [epochs, data[:,4]]

    if out_data is None:
        raise ValueError("Not recognized quantity")

    return [out_data, out_plot_kwargs]


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
        if isinstance(inputs, greedyRoutine):
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

def get_tuning_models(folder, exclude=None, order_col=4):
    import json

    # Collect data:
    info_files = []
    data = []
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
    all_data = np.row_stack(tuple(data))

    # Delete NaN:
    nan_indx = (np.isnan(all_data[:,3])).nonzero()
    all_data = np.delete(all_data,nan_indx,axis=0)

    # Order data:
    all_data = all_data[all_data[:,order_col].argsort()]
    if exclude:
        all_data = all_data[:-exclude]
    model_IDs = all_data[:,0]
    ordered_model_paths = [info_files[0]['path_out']+'model_%d' %model_ID for model_ID in model_IDs]
    return [ordered_model_paths, model_IDs]

def plot_stuff(quantities, inputs, callable, labels=None, callable_args=None, **kwargs):
    from matplotlib.pyplot import cm


    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(quantities, list):
        quantities = [quantities]

    # Check if I have a folder of models:
    tuning = False
    if len(inputs)==1:
        info_logs = [os.path.join(dirpath, f)
           for dirpath, dirnames, files in os.walk(inputs[0])
           for f in files if f.startswith('info-tuning')]
        if len(info_logs)!=0:
            tuning = True
            inputs, labels = get_tuning_models(inputs[0], **kwargs)


    N, Q = len(inputs), len(quantities)
    if labels is None:
        labels = ['']*N
    colors = cm.rainbow(np.linspace(0,1,N))
    lineStyles = ['-', '-.', '--']*10

    plot_kwargs = {}
    plot_kwargs['lineStyles'] = []
    plot_kwargs['colors'] = []
    plot_kwargs['labels'] = []
    additional_kwargs = callable(inputs[0], quantities[0], **callable_args)[1]
    plot_kwargs['xyLabels'] = [additional_kwargs['xyLabels'][0], '']

    xs, ys = [0.]*N*Q, [0.]*N*Q
    for i, input in enumerate(inputs):
        for q, quantity in enumerate(quantities):
            xs[q+i*Q], ys[q+i*Q] = callable(input, quantity, **callable_args)[0]
            plot_kwargs['colors'].append(colors[i])
            plot_kwargs['lineStyles'].append(lineStyles[q])
            plot_kwargs['labels'].append(labels[i] if q==0 else '')


    return xs, ys, plot_kwargs
















