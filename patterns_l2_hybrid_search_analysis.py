import matplotlib.pyplot as plt
import numpy as np
import classification_statistics as cs
import txt_utils as tu
import os
#import errorbars as eb

def SEMcalc(x):
    #Note: x should be an array of the best lambdas for one set of hyperparameters
    N = len(x)
    #Calculate xbar
    mean=meancalc(x)
    
    #Calculate sample variance
    samvar2=1/(N-1)*sum((x-mean)**2)
    #calculate the standard error of the mean (SEM)
    SEM=np.sqrt(samvar2/N)
    
    return SEM

def meancalc(x):
    return sum(x)/len(x)

# Define path to results.txt
#static_hps_ids = ['3913a54a8e78']
static_hps_ids = ['23d311a46ef0', '30b81790d64', '1904f9057076']

results_dir = 'patterns_l2_hybrid_search_results'
results_name = 'results.txt'

# Get all the results.txt files from a given id
cwd = os.getcwd()
id_dirs = []
rootdirs = []
for static_hps_id in static_hps_ids:
    id_dirs.append(f'id-{static_hps_id}')
    rootdirs.append(os.path.join(cwd, results_dir, id_dirs[-1]))

#hyperparameters_to_stats = {}
var3_to_var1_to_list_of_opt_var2 = {}

# For each file:
# 1. Get var3 value
# 2. For each var1:
#    --> Get optimal var2 value by minimising loss
# 3. Update a dictionary of var3_to_var1_to_list_of_opt_var2

# Walk through the directories and subdirectories of rootdir to find the results.txt files
for rootdir in rootdirs:
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if (file == results_name):
                # Open the file 
                txt_name = os.path.join(subdir, file)
                
                # Collect the unique data points in the file
                #hyperparameters_to_stats = {}
                #hyperparameters_to_stats.update(tu.unload_results(txt_name))
                hyperparameters_to_stats = tu.unload_results(txt_name)

                # Collect the static hyperparameters
                static_hyperparameters = tu.unload_static_hyperparameters(txt_name)
                var3 = static_hyperparameters['dropout']
                data_set = static_hyperparameters['dataset']

                # Seperate the data by unique patterns
                var1_to_opt = {}
                for key in hyperparameters_to_stats:
                    stats = hyperparameters_to_stats[key]
                    loss = stats['loss']
                    var1 = key[0]
                    var2 = key[1]

                    if isinstance(var1_to_opt.get(var1, None), type(None)):
                        var1_to_opt[var1] = {}

                    opt = var1_to_opt[var1]
                    min_loss = opt.get('loss', None)
                    if min_loss == None:
                        min_loss = np.inf

                    if loss < min_loss:
                        # Update the optimal var2 and minimum loss
                        var1_to_opt[var1]['var2'] = var2
                        var1_to_opt[var1]['loss'] = loss
                
                if isinstance(var3_to_var1_to_list_of_opt_var2.get(var3, None), type(None)):
                    var3_to_var1_to_list_of_opt_var2[var3] = {}

                for key in var1_to_opt:
                    var1 = key
                    opt = var1_to_opt[var1]
                    opt_var2 = opt['var2']

                    if isinstance(var3_to_var1_to_list_of_opt_var2[var3].get(var1, None), type(None)):
                        var3_to_var1_to_list_of_opt_var2[var3][var1] = []

                    list_of_opt_var2 = var3_to_var1_to_list_of_opt_var2[var3][var1]
                    list_of_opt_var2.append(opt_var2)

                var3_to_var1_to_list_of_opt_var2[var3][var1] = list_of_opt_var2
                
var3_to_plot = {}
for key in var3_to_var1_to_list_of_opt_var2:
    var3 = key
    var1_to_list_of_opt_var2 = var3_to_var1_to_list_of_opt_var2[var3]

    xs = []
    ys = []
    es = []
    for key in var1_to_list_of_opt_var2:
        var1 = key
        list_of_opt_var2 = var1_to_list_of_opt_var2[var1]

        xs.append(var1)
        ys.append(meancalc(np.array(list_of_opt_var2)))
        es.append(SEMcalc(np.array(list_of_opt_var2)))

    var3_to_plot[var3] = {'xs': xs, 'ys': ys, 'es': es}

marker = 'x'
title = f'Random hyperparameter search of L2 and dropout'
x_label = 'Patterns'
y_label = 'Optimal L2 strength'
is_log_plot = True
img_type = 'pdf' # Change to .pdf if generating images for thesis

# Plot the scatter plot with a color bar
def plot_stuff(save_as):
    fig, ax = plt.subplots()

    ax.set_title(title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.set_xlabel(f'log$_{{{10}}}$({x_label})')
    #ax.set_ylabel(f'log$_{{{10}}}$({y_label})')

    #im = ax.scatter(xs, ys, marker=marker)#, c=minimum_losses)
    #im = ax.scatter(xs_2, ys_2, marker=marker, alpha=1/3, label='points tried')#, c=losses)
    var3s = np.array(list(var3_to_plot.keys()))
    var3s[::-1].sort() # Sort var3s into descending orders
    for var3 in var3s:
        plot = var3_to_plot[var3]
        xs = plot['xs']
        ys = plot['ys']
        es = plot['es']

        #invis_col = (0,0,0,0)
        #sc = ax.scatter(xs, ys, marker = 'o', label=f'dropout = {key}')
        #col = sc.get_facecolors()[0].tolist()
        #ax.errorbar(xs, ys, yerr=es, fmt = 'o', color=invis_col, ecolor=col, capsize=4, label=f'with error bars')

        eb = ax.errorbar(xs, ys, yerr=es, fmt = 'o', capsize=4, label=f'dropout = {var3}')
        #connector, caplines, (vertical_lines,) = container.errorbar.lines
        #col = connector.get_color()
        col = eb[0].get_color()

        # Make data logarithmic
        xs_log = np.log10(xs)
        ys_log = np.log10(ys)

        m, c = np.polyfit(xs_log, ys_log, 1)
        f = lambda x : 10**(m*np.log10(x) + c)
        print('dropout', var3)
        print('m', m)
        print('c', c)

        fit = f(np.array(xs))
        plt.plot(xs, fit, '--', c = col, alpha = 0.5)


    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(9, 110)
    ax.set_ylim(1e-5, 1e-1)

    plt.legend(loc = 'upper right')
    plt.savefig(f'{results_dir}/{save_as}')

# Create loss plot
save_as = f'analysis_loss_ids-{static_hps_ids}.{img_type}'
plot_stuff(save_as)

# Create accuracies plot
"""
save_as = f'analysis_accuracies_id-{static_hps_id}.{img_type}'
accuracies = np.dot(accuracies, 100) # Turn into percentage
colour_map = 'gist_rainbow_r'#'rainbow'#'jet'
clim_range = [40, 100]
#clim_range = None
plot_stuff(accuracies, save_as, 'Validation accuracy / %')
"""

plt.show()
