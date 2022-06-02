from cmath import nan
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
#static_hps_ids = ['10f5ef7e1417']

#static_hps_ids = ['626ffedea6c']

#static_hps_ids = ['77e153013f4']

#static_hps_ids = ['72d8278ec4b', '2a33c2b1c82e', '5011b95e557']
#static_hps_ids = ['72d8278ec4b']
#static_hps_ids = ['2a33c2b1c82e']
#static_hps_ids = ['5011b95e557']

#static_hps_ids = ['72d8278ec4b', '2a33c2b1c82e', '346fc9059df4']
#static_hps_ids = ['72d8278ec4b']
#static_hps_ids = ['2a33c2b1c82e']
#static_hps_ids = ['346fc9059df4']

#static_hps_ids = ['10f323679d3b']

#static_hps_ids = ['346fc9059df4', '2a33c2b1c82e', '72d8278ec4b']
#static_hps_ids = ['346fc9059df4'] # 0.5 dropout 1000 epochs
#static_hps_ids = ['2a33c2b1c82e'] # 0.8 1000
#static_hps_ids = ['72d8278ec4b'] # 1.0 1000

static_hps_ids = ['e90e8a81631', '454e254406b', '20b2393a4b51']
#static_hps_ids = ['e90e8a81631'] # 0.5 dropout 4000 epochs
#static_hps_ids = ['454e254406b'] # 0.8 4000
#static_hps_ids = ['20b2393a4b51'] # 1.0 4000

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
var3_to_var1_to_lists_of_opt_var2_and_loss = {}

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
                
                #if isinstance(var3_to_var1_to_list_of_opt_var2.get(var3, None), type(None)):
                if isinstance(var3_to_var1_to_lists_of_opt_var2_and_loss.get(var3, None), type(None)):
                    #var3_to_var1_to_list_of_opt_var2[var3] = {}
                    var3_to_var1_to_lists_of_opt_var2_and_loss[var3] = {}
                     
                for key in var1_to_opt:
                    var1 = key
                    opt = var1_to_opt[var1]
                    opt_var2 = opt['var2']
                    min_loss = opt['loss']

                    #if isinstance(var3_to_var1_to_list_of_opt_var2[var3].get(var1, None), type(None)):
                    if isinstance(var3_to_var1_to_lists_of_opt_var2_and_loss[var3].get(var1, None), type(None)):
                        var3_to_var1_to_lists_of_opt_var2_and_loss[var3][var1] = {'var2' : [], 'loss' : []}
                        #var3_to_var1_to_list_of_opt_var2[var3][var1] = []

                    #list_of_opt_var2 = var3_to_var1_to_list_of_opt_var2[var3][var1]
                    list_of_opt_var2 = var3_to_var1_to_lists_of_opt_var2_and_loss[var3][var1]['var2']
                    list_of_min_loss = var3_to_var1_to_lists_of_opt_var2_and_loss[var3][var1]['loss']
                    list_of_opt_var2.append(opt_var2)
                    list_of_min_loss.append(min_loss)

                #var3_to_var1_to_list_of_opt_var2[var3][var1] = list_of_opt_var2
                var3_to_var1_to_lists_of_opt_var2_and_loss[var3][var1]['var2'] = list_of_opt_var2
                var3_to_var1_to_lists_of_opt_var2_and_loss[var3][var1]['loss'] = list_of_min_loss

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
#for key in var3_to_var1_to_list_of_opt_var2:
for key in var3_to_var1_to_lists_of_opt_var2_and_loss:
    var3 = key
    #var1_to_list_of_opt_var2 = var3_to_var1_to_list_of_opt_var2[var3]
    var1_to_lists_of_opt_var2_and_loss = var3_to_var1_to_lists_of_opt_var2_and_loss[var3]

    xs = []
    ys = []
    es = []

    var3_to_run_to_plot[var3] = {}

    for key in var1_to_lists_of_opt_var2_and_loss:
        var1 = key
        list_of_opt_var2 = var1_to_lists_of_opt_var2_and_loss[var1]['var2']
        list_of_min_loss = var1_to_lists_of_opt_var2_and_loss[var1]['loss']

        xs.append(var1)
        #ys.append(meancalc(np.array(list_of_opt_var2)))
        #print(var1)
        #print(list_of_opt_var2)
        #print(np.log10(np.array(list_of_opt_var2)))

        ys.append(meancalc(np.log10(np.array(list_of_opt_var2))))
        
        #es.append(SEMcalc(np.array(list_of_opt_var2)))
        es.append(SEMcalc(np.log10(np.array(list_of_opt_var2))))

        for run_idx in range(0, len(list_of_opt_var2)):
            if isinstance(var3_to_run_to_plot[var3].get(run_idx, None), type(None)):
                var3_to_run_to_plot[var3][run_idx] = {}

            try:
                xs_run = var3_to_run_to_plot[var3][run_idx]['xs']
                ys_run = var3_to_run_to_plot[var3][run_idx]['ys']
                losses_run = var3_to_run_to_plot[var3][run_idx]['losses']

                xs_run.append(var1)
                ys_run.append(list_of_opt_var2[run_idx])
                losses_run.append(list_of_min_loss[run_idx]) 

                """
                for seed in seed_to_hyperparameters_to_stats:
                    hyperparameters_to_stats = seed_to_hyperparameters_to_stats[seed]
                    try:
                        stats = hyperparameters_to_stats[(int(var1), list_of_opt_var2[run_idx])]
                    except:
                        continue
                loss = stats['loss']
                losses_run.append(loss)
                """

                var3_to_run_to_plot[var3][run_idx]['xs'] = xs_run
                var3_to_run_to_plot[var3][run_idx]['ys'] = ys_run
                var3_to_run_to_plot[var3][run_idx]['losses'] = losses_run
            except:
                # First time
                #print(hyperparameters_to_stats)
                #stats = hyperparameters_to_stats[(int(var1), list_of_opt_var2[run_idx])]
                #loss = stats['loss']
                var3_to_run_to_plot[var3][run_idx] = {'xs': [var1], 'ys': [list_of_opt_var2[run_idx]], 'losses':[list_of_min_loss[run_idx]]}

    print(es)
    es = np.array(es)
    #es = 0.434*es/ys

    print(ys)

    #ys = np.array(ys)
    #ys = 10**ys

    #print(es)

    #es = np.array(es)
    #es = 10**es



    #print(es)

    var3_to_plot[var3] = {'xs': xs, 'ys': ys, 'es': es}

print("min loss =", min(all_losses))
print("max loss =", max(all_losses))

#clim_min = None
#clim_max = None
clim_min = 0.20516624519951462
clim_max = 3.422007943213292 / 4

marker = '.'
#title = f'Random hyperparameter search of L2 and dropout'
x_label = f'Training patterns $N$' #r'Training patterns $N_\mathrm{training}$'
y_label = f'L$_2$-strength $\lambda$'
is_log_plot = True
img_type = 'pdf' # Change to .pdf if generating images for thesis

#var3_to_reg_idx = {1.0: [1, 7], 0.8: [0, 6], 0.5: [2, 5]} #1000
var3_to_reg_idx = {1.0: [3, 8], 0.8: [3, 8], 0.5: [2, 5]} #4000

# Plot the scatter plot with a color bar
def plot_stuff(save_as):
    var3s = np.array(list(var3_to_plot.keys()))
    var3s[::-1].sort() # Sort var3s into descending orders

    #fig, ax = plt.subplots()
    fig, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [6, .75]})
    (ax, ax2) = axes

    plt.subplots_adjust(hspace = 0.075)

    #ax.set_title(title)

    #ax.set_xlabel(x_label)
    ax2.set_xlabel(x_label)
    #ax.set_ylabel(y_label)
    fig.text(0.02, 0.5, y_label, va='center', rotation='vertical')
    #ax2.set_ylabel(y_label)

    colour_map = 'gist_rainbow'#'gnuplot2'#'turbo'

    for var3 in var3s:
        plot = var3_to_plot[var3]
        xs = plot['xs']
        ys = plot['ys']
        es = plot['es']

        print("ys,", ys)
        
        #invis_col = (0,0,0,0)
        #sc = ax.scatter(xs, ys, marker = 'o', label=f'dropout = {key}')
        #col = sc.get_facecolors()[0].tolist()
        #ax.errorbar(xs, ys, yerr=es, fmt = 'o', color=invis_col, ecolor=col, capsize=4, label=f'with error bars')

        #ys = np.array(ys)
        #ys = np.log10(ys)
        #print('ys:', ys)

        es = np.array(es)
        np.nan_to_num(es, copy=False)
        if len(var3s) == 1:
            col = 'k'

            eb = ax.errorbar(xs, ys, yerr=es, fmt = 'o', capsize=4, c=col, label=f'$P$ = {(1-var3):.1f}')
            eb = ax2.errorbar(xs, 10**np.array(ys), yerr=es, fmt = 'o', capsize=4, c=col, label=f'$P$ = {(1-var3):.1f}')
        else:
            eb = ax.errorbar(xs, ys, yerr=es, fmt = 'o', capsize=4, label=f'$P$ = {(1-var3):.1f}') 
            eb = ax2.errorbar(xs, 10**np.array(ys), yerr=es, fmt = 'o', capsize=4, label=f'$P$ = {(1-var3):.1f}')
            col = eb[0].get_color()

        # Linear regression...
        #xs_reg = xs[2:6]
        xs_reg = xs[var3_to_reg_idx[var3][0]:var3_to_reg_idx[var3][1]]
        #ys_reg = ys[2:6]
        ys_reg = ys[var3_to_reg_idx[var3][0]:var3_to_reg_idx[var3][1]]
        
        # Make data logarithmic
        xs_log = np.log10(xs_reg)
        #print(ys_reg)
        #xs_log = xs_reg
        ys_log = ys_reg

        print(ys_log)

        #ys_log = np.log10(ys_reg)

        m, c = np.polyfit(xs_log, ys_log, 1)
        f = lambda x : 10**(m*np.log10(x) + c)
        print('dropout', 1-var3)
        print('m', m)
        print('c', c)
        print()

        fit = f(np.array(xs_reg))
        fit = np.log10(fit)
        print(fit)
        ax.plot(xs_reg, fit, '--', c = col, alpha = 1.0)
        #ax2.plot(xs_reg, fit, '--', c = col, alpha = 1.0)

    ax.set_xscale('log')
    ax2.set_xscale('log')

    ax.set_yscale('linear')
    #ax.set_yscale('log')#, nonposy="clip")
    #ax.set_yscale('symlog', linthresh=1e-5)
    #ax.set_yscale('log')
    #ax2.set_yscale('linear')
    ax2.set_yscale('symlog', linthresh=.5e-5)#, nonposy="clip")

    #ax.set_xlim(None, 1.5e+3)
    #ax2.set_xlim(None, 1.5e+3)

    #ax.set_ylim(1e-6, 1e-1)
    #ax.set_ylim(.5e-5, 1)
    #ax2.set_ylim(-.5e-5, .5e-5)
    ax.set_ylim(-5.30102999566, 0)
    ax2.set_ylim(-.5e-5, .5e-5)

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', which='both', labeltop=False, bottom = False, top = False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    
    ax2.set_yticks([0])

    d = .01  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=0.8)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - 8*d, 1 + 8*d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - 8*d, 1 + 8*d), **kwargs)  # bottom-right diagonal

    ax.legend(loc=1, title='Drop-rate:')
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
