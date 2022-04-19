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
    try:
        samvar2 = 1/(N-1)*sum((x-mean)**2)
    except:
        # In case of division by zero (...? all the same point?)
        samvar2 = 0
    #calculate the standard error of the mean (SEM)
    SEM=np.sqrt(samvar2/N)
    
    return SEM

def meancalc(x):
    return sum(x)/len(x)

# Define path to results.txt
#static_hps_ids = ['77e153013f4']

static_hps_ids = ['72d8278ec4b', '2a33c2b1c82e', '5011b95e557']
#static_hps_ids = ['72d8278ec4b']
#static_hps_ids = ['2a33c2b1c82e']
#static_hps_ids = ['5011b95e557']

#static_hps_ids = ['25b5feffb9c7', '2b3b4ce3670d', '3caceb59003a']
#static_hps_ids = ['25b5feffb9c7']
#static_hps_ids = ['2b3b4ce3670d']
#static_hps_ids = ['3caceb59003a']

results_dir = 'patterns_l2_hybrid_search_results'
results_name = 'results.txt'

# Get all the results.txt files from a given id
cwd = os.getcwd()
id_dirs = []
rootdirs = []
for static_hps_id in static_hps_ids:
    id_dirs.append(f'id-{static_hps_id}')
    rootdirs.append(os.path.join(cwd, results_dir, id_dirs[-1]))

seed_to_hyperparameters_to_stats = {}
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
                hyperparameters_to_stats = tu.unload_results(txt_name)
                hyperparameters_to_stats_2 = {}

                # Double the number of patterns...
                for key in hyperparameters_to_stats:
                    patterns = key[0] * 2 # Double the number of patterns, since they refer to the number of training patterns of EACH distribution.
                    l2 = key[1]

                    key_2 = (patterns, l2)
                    hyperparameters_to_stats_2[key_2] = hyperparameters_to_stats[key]

                hyperparameters_to_stats = hyperparameters_to_stats_2

                seeds = tu.unload_seeds(txt_name)
                search_seed = seeds['search']
                if isinstance(seed_to_hyperparameters_to_stats.get(search_seed, None), type(None)):
                        seed_to_hyperparameters_to_stats[search_seed] = {}
                seed_to_hyperparameters_to_stats[search_seed].update(hyperparameters_to_stats)
                #all_hyperparameters_to_stats.update(hyperparameters_to_stats)

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

#Collect a plot of all the chosen points
all_plot = {}
all_xs = []
all_ys = []
all_losses = []
test_seed = None
for key in seed_to_hyperparameters_to_stats:
    seed = key
    if not isinstance(test_seed, type(None)):
        if seed != test_seed:
            continue

    hyperparameters_to_stats = seed_to_hyperparameters_to_stats[seed]
    for key in hyperparameters_to_stats:
        x, y = key
        stats = hyperparameters_to_stats[key]
        loss = stats['loss']
        all_xs.append(x)
        all_ys.append(y)
        all_losses.append(loss)

# Collect the optimal plots for each dropout
var3_to_plot = {}
var3_to_run_to_plot = {}
for key in var3_to_var1_to_list_of_opt_var2:
    var3 = key
    var1_to_list_of_opt_var2 = var3_to_var1_to_list_of_opt_var2[var3]

    xs = []
    ys = []
    es = []
    
    var3_to_run_to_plot[var3] = {}

    for key in var1_to_list_of_opt_var2:
        var1 = key
        list_of_opt_var2 = var1_to_list_of_opt_var2[var1]

        xs.append(var1)
        ys.append(meancalc(np.array(list_of_opt_var2)))
        es.append(SEMcalc(np.array(list_of_opt_var2)))

        for run_idx in range(0, len(list_of_opt_var2)):
            if isinstance(var3_to_run_to_plot[var3].get(run_idx, None), type(None)):
                var3_to_run_to_plot[var3][run_idx] = {}

            try:
                xs_run = var3_to_run_to_plot[var3][run_idx]['xs']
                ys_run = var3_to_run_to_plot[var3][run_idx]['ys']
                #losses_run = var3_to_run_to_plot[var3][run_idx]['losses']

                xs_run.append(var1)
                ys_run.append(list_of_opt_var2[run_idx])

                """
                for seed in seed_to_hyperparameters_to_stats:
                    hyperparameters_to_stats = seed_to_hyperparameters_to_stats[seed] 
                    if  hyperparameters_to_stats
                stats = hyperparameters_to_stats[(int(var1), list_of_opt_var2[run_idx])]
                loss = stats['loss']
                losses_run.append(loss)
                """

                var3_to_run_to_plot[var3][run_idx]['xs'] = xs_run
                var3_to_run_to_plot[var3][run_idx]['ys'] = ys_run
                #var3_to_run_to_plot[var3][run_idx]['losses'] = losses_run
            except:
                # First time
                #print(hyperparameters_to_stats)
                #stats = hyperparameters_to_stats[(int(var1), list_of_opt_var2[run_idx])]
                #loss = stats['loss']
                var3_to_run_to_plot[var3][run_idx] = {'xs': [var1], 'ys': [list_of_opt_var2[run_idx]]}#, 'losses':[loss]}

    var3_to_plot[var3] = {'xs': xs, 'ys': ys, 'es': es}



marker = '.'
title = f'Random hyperparameter search of L2 and dropout'
x_label = 'Patterns'
y_label = 'Optimal L2 strength'
is_log_plot = True
img_type = 'pdf' # Change to .pdf if generating images for thesis

# Plot the scatter plot with a color bar
def plot_stuff(save_as):
    var3s = np.array(list(var3_to_plot.keys()))
    var3s[::-1].sort() # Sort var3s into descending orders

    fig, ax = plt.subplots()

    ax.set_title(title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    colour_map = 'gist_rainbow'#'gnuplot2'#'turbo'

    if len(var3s) == 1:
        im = ax.scatter(all_xs, all_ys, marker=marker, alpha=1, c=all_losses, cmap=colour_map)
        cb = fig.colorbar(im, ax=ax)
    else:
        im = ax.scatter(all_xs, all_ys, marker=marker, c='k', alpha=0.05)

    for var3 in var3s:
        plot = var3_to_plot[var3]
        xs = plot['xs']
        ys = plot['ys']
        es = plot['es']

        if len(var3s) == 1:
            run_to_plot = var3_to_run_to_plot[var3]
            for run_idx in run_to_plot:
                # Plot the individual samples
                plot_run = run_to_plot[run_idx]
                xs_run = plot_run['xs']
                ys_run = plot_run['ys']

                #1. get ys_run index from all_ys
                loss_run = []
                for y in ys_run:
                    idx = all_ys.index(y)
                    
                    #2. get all_loss value from all_losses
                    loss = all_losses[idx]
                    loss_run.append(loss)

                im = ax.scatter(xs_run, ys_run, marker='o', alpha=1, c=loss_run, cmap=colour_map)
                im.set_clim(min(all_losses), max(all_losses))

                plt.plot(xs_run, ys_run, '--', alpha = 0.5)
            

        #invis_col = (0,0,0,0)
        #sc = ax.scatter(xs, ys, marker = 'o', label=f'dropout = {key}')
        #col = sc.get_facecolors()[0].tolist()
        #ax.errorbar(xs, ys, yerr=es, fmt = 'o', color=invis_col, ecolor=col, capsize=4, label=f'with error bars')


        if len(var3s) == 1:
            col = 'k'
            eb = ax.errorbar(xs, ys, yerr=es, fmt = 'o', capsize=4, c=col, label=f'dropout = {var3}')
        else:
            eb = ax.errorbar(xs, ys, yerr=es, fmt = 'o', capsize=4, label=f'dropout = {var3}')
            col = eb[0].get_color()

        # Make data logarithmic
        xs_log = np.log10(xs)
        ys_log = np.log10(ys)

        m, c = np.polyfit(xs_log, ys_log, 1)
        f = lambda x : 10**(m*np.log10(x) + c)
        print('dropout', var3)
        print('m', m)
        print('c', c)
        print()

        fit = f(np.array(xs))
        plt.plot(xs, fit, '--', c = col, alpha = 1.0)

    ax.set_xscale('log')

    ax.set_yscale('log')
    #ax.set_yscale('symlog', linthresh=1e-6)

    ax.set_ylim(1e-6, 1e-1)
    #ax.set_ylim(-.25e-6, 1e-1)

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
