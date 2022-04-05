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
static_hps_ids = ['3913a54a8e78']

results_dir = 'patterns_l2_hybrid_search_results'
results_name = 'results.txt'

# Get all the results.txt files from a given id
cwd = os.getcwd()
id_dirs = []
rootdirs = []
for static_hps_id in static_hps_ids:
    id_dirs.append(f'id-{static_hps_id}')
    rootdirs.append(os.path.join(cwd, results_dir, id_dirs[-1]))

hyperparameters_to_stats = {}

temp_list_optimal_l2_loss = []
temp_list_optimal_l2 = []
patterns_to_optimal_l2 = {}
patterns_to_optimal_l2_loss = {}

dropout_to_patterns_to_optimal_l2 = {}

# Walk through the directories and subdirectories of rootdir to find the results.txt files
for rootdir in rootdirs:
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if (file == results_name):
                # Open the file 
                txt_name = os.path.join(subdir, file)
                
                # Collect the unique data points in the file
                hyperparameters_to_stats.update(tu.unload_results(txt_name))

                # Collect the static hyperparameters
                static_hyperparameters = tu.unload_static_hyperparameters(txt_name)
                dropout = static_hyperparameters['dropout']
                data_set = static_hyperparameters['dataset']

                patterns_to_l2 = {}
                temp_list = []
                # Seperate the data by unique patterns
                for key in tu.unload_results(txt_name):
                    stats = tu.unload_results(txt_name)[key]
                    try:   
                        temp_list = patterns_to_l2[key[0]]
                    except:
                        temp_list = []
                    temp_list.append({'l2':key[1], 'loss':stats['loss']})
                    patterns_to_l2[int(key[0])] = temp_list

                unique_patterns = []
                optimal_l2s = []
                minimum_losses = []
                # Get the minimum loss for each unique patterns
                for key in patterns_to_l2:
                    optimal_l2 = np.inf
                    minimum_loss = np.inf
                    for i in range(0, len(patterns_to_l2[key])):
                        l2 = patterns_to_l2[key][i]['l2']
                        loss = patterns_to_l2[key][i]['loss']
                        if loss < minimum_loss:
                            optimal_l2 = l2
                            minimum_loss = loss

                    unique_patterns.append(key)
                    optimal_l2s.append(optimal_l2)
                    minimum_losses.append(minimum_loss)
                
                for i in range(0, len(unique_patterns)):
                    try:
                        temp_list_optimal_l2_loss = patterns_to_optimal_l2_loss[unique_patterns[i]]
                        temp_list_optimal_l2 = patterns_to_optimal_l2[unique_patterns[i]]
                    except:
                        temp_list_optimal_l2_loss = []
                        temp_list_optimal_l2 = []
                    temp_list_optimal_l2_loss.append({'l2':optimal_l2s[i], 'loss':minimum_losses[i]})
                    temp_list_optimal_l2.append(optimal_l2s[i])
                    patterns_to_optimal_l2_loss[unique_patterns[i]] = temp_list_optimal_l2_loss
                    patterns_to_optimal_l2[unique_patterns[i]] = temp_list_optimal_l2

                dropout_to_patterns_to_optimal_l2[dropout] = patterns_to_optimal_l2

# Seperate the dropouts, l2s and losses to check which l2 values were tested
patterns = []
l2s = []
losses = []
accuracies = []
for key in hyperparameters_to_stats:
    patterns.append(key[0])
    l2s.append(key[1])
    stats = hyperparameters_to_stats[key]
    losses.append(stats['loss'])
    # Determine accuracy from the confusion matrix
    val_stats = cs.calculate_stats(stats['cm']) # Calculate statistics from the confusion matrix
    fin_val_acc = val_stats[0]['advanced']['acc'] # Get validation accuracy
    accuracies.append(fin_val_acc)

# Fixed plotting stuff
xs_2 = patterns
ys_2 = l2s

xs = unique_patterns

dropouts = []
patterns_to_optimal_l2s = []

dropout_to_ys = {}
dropout_to_errorbars = {}

for key in dropout_to_patterns_to_optimal_l2:
    dropouts.append(key)

    patterns_to_optimal_l2 = dropout_to_patterns_to_optimal_l2[key]

    ys = [meancalc(np.array(patterns_to_optimal_l2[point])) for point in xs]
    error_bars = [SEMcalc(np.array(patterns_to_optimal_l2[point])) for point in xs]

    dropout_to_ys[key] = ys
    dropout_to_errorbars[key] = error_bars

marker = 'x'
title = f'Random hyperparameter search of L2 and dropout\n({data_set})'
x_label = 'Patterns'
y_label = 'L2 strength'
is_log_plot = True
img_type = 'pdf' # Change to .pdf if generating images for thesis

# Plot the scatter plot with a color bar
def plot_stuff(save_as):
    fig, ax = plt.subplots()

    ax.set_title(title)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(f'log$_{{{10}}}$({x_label})')
    ax.set_ylabel(f'log$_{{{10}}}$({y_label})')

    #im = ax.scatter(xs, ys, marker=marker)#, c=minimum_losses)
    im = ax.scatter(xs_2, ys_2, marker=marker, alpha=1/3)#, c=losses)
    for key in dropout_to_ys:
        transparent_colour = (1,1,1,0)
        sc = ax.scatter(xs, dropout_to_ys[key], marker = 'o', label=f'dropout = {key}')
        col = sc.get_facecolors()[0].tolist()
        ax.errorbar(xs, dropout_to_ys[key], yerr=dropout_to_errorbars[key], fmt = 'o', color=transparent_colour, ecolor=col, capsize=4, label=f'with error bars')

    ax.set_xlim(9, 1100)
    ax.set_ylim(1e-6, 1)

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
