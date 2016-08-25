import matplotlib

import matplotlib.pyplot as plt

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

    # Readjust mean image and variance:
    mean_train = np.reshape([[[ 73.15835921 , 82.90891754, 72.39239876]]],(3,1,1)) /255.
    var_train = np.reshape([[[ 2237.79144756,  2326.24575092, 2256.68620499]]],(3,1,1)) / 255.**2
    images = np.add(images, mean_train)
    images = np.multiply(images, np.sqrt(var_train))

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
        kargs["labels"] = [""]*20
    if not "legend" in kargs:
        kargs["legend"] = "best"
    if not "colors" in kargs:
        kargs["colors"] = ['r','b','g','y','m','k']*20

    # For ellipses:
    if type_of_plot=="ellipses":
        if not "ticks_size" in kargs:
            kargs["ticks_size"] = 9
        if not "label_size" in kargs:
            kargs["label_size"] = kargs["ticks_size"]
        if not "opacities" in kargs:
            kargs["opacities"] = [0.3]*10

    # For line plots (particularly spectra...)
    elif type_of_plot=="linePlot":
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
    return kargs

def plot_fcts(axis, x, ys, **plot_kargs):
    plot_kargs = check_matplot_arguments("linePlot",**plot_kargs)

    matplotlib.rcParams['text.usetex'] = True
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

    matplotlib.rcParams['text.usetex'] = True
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

    matplotlib.rcParams['text.usetex'] = True
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



