import os
import numpy as np
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import json

from lasagne.layers import get_all_param_values

from nolearn.lasagne import NeuralNet


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


def plot_predict_proba(net, X, y, pdf_name, mistakes_plot=False, images_plot=False, prediction_plot=False):
    '''
    Don't put the extension in the pdf_name...
    '''
    predict_proba = net.predict_proba(X)
    fig = plot_GrTruth(predict_proba[:,1,:,:])
    fig.savefig(pdf_name+'_predProb.pdf')
    fig = plot_GrTruth(y)
    fig.savefig(pdf_name+'_GrTr.pdf')
    if mistakes_plot:
        fig = plot_GrTruth(np.abs(predict_proba[:,1,:,:]-y))
        fig.savefig(pdf_name+'_mistakes.pdf')
    if images_plot:
        fig = plot_images(X)
        fig.savefig(pdf_name+'_inputImages.pdf')
    if prediction_plot:
        predict = predict_proba.argmax(axis=1)
        fig = plot_GrTruth(predict)
        fig.savefig(pdf_name+'_pred.pdf')



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



# ###############################################
#   ADVANCED PLOTTING:
# ###############################################


def get_model_data(inputs, quantity):
    '''
    Inputs:
        - input should contain a model or a path
    '''
    # Import files:
    import six
    if isinstance(inputs, NeuralNet):
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
    ordered_model_paths = [info_files[0]['path_out']+'model_%d/' %model_ID for model_ID in model_IDs]
    return [ordered_model_paths, model_IDs]

def plot_stuff(quantities, inputs, callable, labels=None, callable_args={}, **kwargs):
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
    lineStyles = ['-', '--', '--']*10

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
        data.append(np.loadtxt(folder+'log-%d.txt' %tun_ID))
        with open(folder+'info-tuning_%d.txt' %tun_ID) as info_file:
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

        if x_max:
            ax.set_xlim([x_min,x_max])
        if y_max:
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





# def compare_stuff(folder, collect_function, quantity, tun_ID=None, exclude=None, plot_kwargs={}):
#     '''
#     Take all the models trained in a tuning procedure and plot some quantity (e.g.
#         the loss) for every model.
#     '''
#     # Collect data:
#     info_files = []
#     data = []
#     if tun_ID:
#         data.append(np.loadtxt(folder+'/log-%d.txt' %tun_ID))
#         with open(folder+'/info-tuning_%d.txt' %tun_ID) as info_file:
#             info_files.append(json.load(info_file))
#     else:
#         # IMPLEMENT REGEX!
#         log_list = [os.path.join(dirpath, f)
#             for dirpath, dirnames, files in os.walk(folder)
#             for f in files if f.startswith('log-')]
#         info_list = [os.path.join(dirpath, f)
#             for dirpath, dirnames, files in os.walk(folder)
#             for f in files if f.startswith('info-tuning_')]
#         for info_name in info_list:
#             with open(info_name) as info_file:
#                 info_files.append(json.load(info_file))
#         data = [np.loadtxt(filename) for filename in log_list]

#     # # Merge data:
#     # x_min = min([info_files[i]['hyperparams'][0][1] for i in range(len(info_files))])
#     # x_max = max([info_files[i]['hyperparams'][0][2] for i in range(len(info_files))])
#     # y_min = min([info_files[i]['hyperparams'][1][1] for i in range(len(info_files))])
#     # y_max = max([info_files[i]['hyperparams'][1][2] for i in range(len(info_files))])
#     # x_info = info_files[0]['hyperparams'][0]
#     # y_info = info_files[0]['hyperparams'][1]
#     all_data = np.row_stack((data[i] for i in range(len(data))))

#     # Delete NaN:
#     # print all_data.shape
#     nan_indx = (np.isnan(all_data[:,3])).nonzero()
#     all_data = np.delete(all_data,nan_indx,axis=0)
#     # print nan_indx
#     # print all_data.shape

#     # Order data:
#     all_data = all_data[all_data[:,3].argsort()]
#     if exclude:
#         all_data = all_data[:-exclude]
#     # input_x = all_data[:,1]
#     # input_y = all_data[:,2]
#     models = all_data[:,0]

#     outputs = [ collect_function(model, quantity, info_files[0]['path_out']) for model in models]
#     N = all_data.shape[0]

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     vis.plot_fcts_PRO(ax, [output[0] for output in outputs], [output[1] for output in outputs], labels=models.astype(np.uint32), **plot_kwargs)
#     # vis.plot_fcts_show(axis, x, ys,     )
#     name = quantity+'-'+tun_ID+'.pdf' if tun_ID else quantity+'-ALL'
#     if exclude:
#         name+='-%d' %exclude
#     fig.savefig(folder+'/'+name+'.pdf')






# def analyse_model(folder, collect_function, quantities, model_ID, plot_kwargs={}):
#     '''
#     Take all the models trained in a tuning procedure and plot some quantity (e.g.
#         the loss) for every model.
#     '''
#     outputs = [ collect_function(model_ID, quantity, folder+'/') for quantity in quantities]

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     vis.plot_fcts_PRO(ax, [output[0] for output in outputs], [output[1] for output in outputs], labels=quantities, **plot_kwargs)
#     # vis.plot_fcts_show(axis, x, ys,     )
#     fig.savefig(folder+'/model_%d/plot_%s.pdf' %(model_ID, quantities[0]))





