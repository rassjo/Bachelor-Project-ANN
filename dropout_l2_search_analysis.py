import matplotlib.pyplot as plt
import numpy as np
import classification_statistics as cs
import txt_utils as tu
import os
import matplotlib.tri as tri

# Define path to results.txt
static_hps_id = '18a0a2942e7e'
#static_hps_id = '1464aef76990'
#static_hps_id = '1a77c2bb16b9' 
#static_hps_id = '1d0ffeda4835' 
#static_hps_id = '32921c5284ae' 

results_dir = 'dropout_l2_random_search_results'
id_dir = f'id-{static_hps_id}'
results_name = 'results.txt'

# Get all the results.txt files from a given id
cwd = os.getcwd()
rootdir = os.path.join(cwd, results_dir, id_dir)

hyperparameters_to_stats = {}

# Walk through the directories and subdirectories of rootdir to find the results.txt files
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if (file == results_name):
            # Open the file 
            txt_name = os.path.join(subdir, file)
            
            # Collect the unique data points in the file
            hyperparameters_to_stats.update(tu.unload_results(txt_name))

            # Collect the static hyperparameters
            static_hyperparameters = tu.unload_static_hyperparameters(txt_name)
            data_set = static_hyperparameters['dataset']
            hidden = static_hyperparameters['hidden']

# Seperate the dropouts, l2s and losses for intuitive plotting 
dropouts = []
l2s = []
losses = []
accuracies = []
for key in hyperparameters_to_stats:
    dropouts.append(key[0])
    l2s.append(key[1])
    stats = hyperparameters_to_stats[key]
    losses.append(stats['loss'])
    # Determine accuracy from the confusion matrix
    val_stats = cs.calculate_stats(stats['cm']) # Calculate statistics from the confusion matrix
    fin_val_acc = val_stats[0]['advanced']['acc'] # Get validation accuracy
    accuracies.append(fin_val_acc)

# Fixed plotting stuff
xs = dropouts
ys = l2s
marker = 'x'
title = f'Random hyperparameter search of L2 and dropout\n({hidden} hidden nodes)'
x_label = 'Dropout rate'
y_label = 'L2 strength'
is_log_plot = True
img_type = 'pdf' # Change to .pdf if generating images for thesis

# Plot the scatter plot with a color bar
def plot_stuff(vals, save_as, colour_bar_label, colour_map, clim_range):
    fig, ax = plt.subplots()
    im = ax.scatter(xs, np.log10(ys), marker=marker, c=vals, cmap=colour_map)
    if isinstance(clim_range, type(None)):
        im.set_clim(min(vals), max(vals)) # This is great, but lacks consistency between plots
    else:
        im.set_clim(clim_range[0], clim_range[1]) # Consistent, for multiple plots
    fig.colorbar(im, ax=ax, label=colour_bar_label)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_xlim(0.5, 1.0)
    if (is_log_plot):
        #ax.set_yscale('log')
        #ax.set_ylim(1e-6, 1)
        ax.set_ylabel(f'log$_{{{10}}}$({y_label})')
    else:
        ax.set_ylim(0, 1)
        ax.set_ylabel(y_label)
    plt.savefig(f'{results_dir}/{id_dir}/{save_as}')

def tri_plot(vals, save_as, colour_bar_label, colour_map, clim_range):
    print(xs)
    print(np.log10(ys))

    triang = tri.Triangulation(xs, np.log10(ys))

    #refiner = tri.UniformTriRefiner(triang)
    #tri_refi, z_test_refi = refiner.refine_field(vals, subdiv=3)
    #tri.TriAnalyzer(triang)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    im1 = ax.tricontourf(triang, vals, cmap=colour_map)
    #im1 = ax.tricontourf(tri_refi, z_test_refi, cmap=colour_map)
    #im2 = ax.triplot(triang)

    if isinstance(clim_range, type(None)):
        im1.set_clim(min(vals), max(vals)) # This is great, but lacks consistency between plots
    else:
        im1.set_clim(clim_range[0], clim_range[1]) # Consistent, for multiple plots
    fig.colorbar(im1, ax=ax, label=colour_bar_label)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_xlim(0.5, 1.0)
    if (is_log_plot):
        #ax.set_yscale('log')
        #ax.set_ylim(1e-6, 1)
        ax.set_ylabel(f'log$_{{{10}}}$({y_label})')
    else:
        ax.set_ylim(0, 1)
        ax.set_ylabel(y_label)
    plt.savefig(f'{results_dir}/{id_dir}/{save_as}')
    
# Create loss plot
save_as = f'analysis_loss_id-{static_hps_id}.{img_type}'
colour_map = 'gist_rainbow'
clim_range = [0.225, 0.7] 
#clim_range = None

if clim_range != None:
    if min(losses) < clim_range[0]:
        print(f"Warning: Minimum loss {min(losses)} is below clim range.")
    if max(losses) > clim_range[1]:
        print(f"Warning: Maximum loss {max(losses)} is above clim range.")

plot_stuff(losses, save_as, 'Validation loss', colour_map, clim_range)

print(np.log10(ys))

save_as = f'tri_analysis_loss_id-{static_hps_id}.{img_type}'
tri_plot(losses, save_as, 'Validation loss', colour_map, clim_range)
# Try using https://matplotlib.org/stable/api/tri_api.html#matplotlib.tri.TriAnalyzer
# TriAnalyzer or TriRefiner or UniformTriRefiner



# Create accuracies plot
#save_as = f'analysis_accuracies_id-{static_hps_id}.{img_type}'
#accuracies = np.dot(accuracies, 100) # Turn into percentage
#colour_map = 'gist_rainbow_r'#'rainbow'#'jet'
#clim_range = [40, 100]
#clim_range = None
#plot_stuff(accuracies, save_as, 'Validation accuracy / %', colour_map, clim_range)

plt.show()
