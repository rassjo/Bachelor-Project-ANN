import matplotlib.pyplot as plt
import numpy as np
import classification_statistics as cs
import txt_utils as tu
import os
import matplotlib.tri as tri

# Define path to results.txt


# dropout & l2:
static_hps_id = '384147b38d9f'
#static_hps_id = '1d0ffeda4835' # extra hidden nodes
#static_hps_id = '294b8b77c209' # low learning rate
#static_hps_id = '18616fcaad7b' # 2d dataset
#static_hps_id = '31ae0c6e5cc3' # extra low learning rate
var1_name = 'dropout'
var1_label = 'Dropout rate'
var2_name = 'l2'
var2_label = 'L2 strength'
#clim_range = [0.225, 0.7] 
clim_range = None
x_lim = [0, 1]
y_lim = [1e-6, 1]
is_x_log = False
is_y_log = True

"""
# patterns & l2:
static_hps_id = '3913a54a8e78'
var1_name = 'patterns'
var1_label = 'Patterns'
var2_name = 'l2'
var2_label = 'L2 strength'
clim_range = None
x_lim = [1e+1, 1e+3]
y_lim = [1e-6, 1]
is_x_log = True
is_y_log = True
"""

"""
# patterns & dropout:
static_hps_id = '298e23193822'
var1_name = 'patterns'
var1_label = 'Patterns'
var2_name = 'dropout'
var2_label = 'Dropout rate'
clim_range = None
x_lim = [1e+1, 1e+3]
y_lim = [0.5, 1]
is_x_log = True
is_y_log = False
"""

results_dir = f'{var1_name}_{var2_name}_random_search_results'
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
var1s = []
var2s = []
losses = []
accuracies = []
for key in hyperparameters_to_stats:
    var1s.append(key[0])
    var2s.append(key[1])
    stats = hyperparameters_to_stats[key]
    losses.append(stats['loss'])
    # Determine accuracy from the confusion matrix
    val_stats = cs.calculate_stats(stats['cm']) # Calculate statistics from the confusion matrix
    fin_val_acc = val_stats[0]['advanced']['acc'] # Get validation accuracy
    accuracies.append(fin_val_acc)

# Fixed plotting stuff
marker = 'x'
static_hps_keys = list(static_hyperparameters.keys())
static_hps_vals = list(static_hyperparameters.values())
static_hps_str = ''
for i in range(0, len(static_hps_keys)):
    static_hp_key = static_hps_keys[i]
    static_hp_val = static_hps_vals[i]

    curr = f'{static_hp_key}: {static_hp_val}'

    if i == 0 and i == len(static_hps_keys)-1:
        static_hps_str = f'({curr})'
    else:
        if i == 0:
            static_hps_str = f'({curr} '
        elif i == len(static_hps_keys)-1:
            static_hps_str = f'{static_hps_str}, {curr})'
        else:
            static_hps_str = f'{static_hps_str}, {curr}'
#for key in static_hyperparameters:
#    static_hp = static_hyperparameters[key]
#    static_hps_str = f"{static_hps_str}, {key.capitalize()}: {static_hp}"
#    if 
title = f'Random hyperparameter search of {var2_name.capitalize()} and {var1_name.capitalize()}\n{static_hps_str}'
x_label = f'{var1_label}'
y_label = f'{var2_label}'
img_type = 'pdf' # Change to .pdf if generating images for thesis

# Plot the scatter plot with a color bar
def plot_stuff(xs, ys, vals, save_as, colour_bar_label, colour_map, clim_range, x_lim, y_lim, is_x_log, is_y_log):
    fig, ax = plt.subplots()
    im = ax.scatter(xs, ys, marker=marker, c=vals, cmap=colour_map)
    if isinstance(clim_range, type(None)):
        im.set_clim(min(vals), max(vals)) # This is great, but lacks consistency between plots
    else:
        im.set_clim(clim_range[0], clim_range[1]) # Consistent, for multiple plots
    fig.colorbar(im, ax=ax, label=colour_bar_label)
    ax.set_title(title)
    ax.set_xlabel(x_label)

    if (is_x_log):
        ax.set_xscale('log')
        ax.set_ylabel(x_label)
        #ax.set_xlabel(f'log$_{{{10}}}$({x_label})')
    else:
        ax.set_xlabel(x_label)

    if (is_y_log):
        ax.set_yscale('log')
        ax.set_ylabel(y_label)
        #ax.set_ylabel(f'log$_{{{10}}}$({y_label})')
    else:
        ax.set_ylabel(y_label)

    if not isinstance(x_lim, type(None)):
        ax.set_xlim(x_lim[0], x_lim[1])
    if not isinstance(y_lim, type(None)):
        ax.set_ylim(y_lim[0], y_lim[1])    
    plt.savefig(f'{results_dir}/{id_dir}/{save_as}')


def tri_plot(xs, ys, vals, save_as, colour_bar_label, colour_map, clim_range):
    triang = tri.Triangulation(xs, ys)
    #triang = tri.Triangulation(xs_ys[:, 0], xs_ys[:, 1])

    #refiner = tri.UniformTriRefiner(triang)
    #tri_refi, z_test_refi = refiner.refine_field(vals, subdiv=3)
    #tri.TriAnalyzer(triang)

    xi, yi = np.meshgrid(np.linspace(min(xs), max(xs), 100), np.linspace(min(ys), max(ys), 100))
    interp_cubic_geom = tri.CubicTriInterpolator(triang, vals, kind='geom')
    zi_cubic_geom = interp_cubic_geom(xi, yi)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    im1 = ax.contourf(xi, yi, zi_cubic_geom)
    #im1 = ax.tricontourf(triang, vals, cmap=colour_map)
    #im1 = ax.tricontourf(tri_refi, z_test_refi, cmap=colour_map)
    #im2 = ax.triplot(triang)

    if isinstance(clim_range, type(None)):
        im1.set_clim(min(vals), max(vals)) # This is great, but lacks consistency between plots
    else:
        im1.set_clim(clim_range[0], clim_range[1]) # Consistent, for multiple plots
    fig.colorbar(im1, ax=ax, label=colour_bar_label)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    #ax.set_xlim(0.5, 1.0)
    if (is_y_log):
        #ax.set_yscale('log')
        #ax.set_ylim(1e-6, 1)
        ax.set_ylabel(f'log$_{{{10}}}$({y_label})')
    else:
        #ax.set_ylim(0, 1)
        ax.set_ylabel(y_label)
    plt.savefig(f'{results_dir}/{id_dir}/{save_as}')
    
# Create loss plot
save_as = f'analysis_loss_id-{static_hps_id}.{img_type}'
colour_map = 'gist_rainbow'

if clim_range != None:
    if min(losses) < clim_range[0]:
        print(f"Warning: Minimum loss {min(losses)} is below clim range.")
    if max(losses) > clim_range[1]:
        print(f"Warning: Maximum loss {max(losses)} is above clim range.")

plot_stuff(var1s, var2s, losses, save_as, 'Validation loss', colour_map, clim_range, x_lim, y_lim, is_x_log, is_y_log)

""" 
# Shaded plot

# Standardise data
xs = var1s
xs_norm = np.linalg.norm(xs)
xs = var1s/xs_norm

ys = var2s
ys = np.log10(ys)
ys_norm = np.linalg.norm(ys)
ys = ys/ys_norm

save_as = f'tri_analysis_loss_id-{static_hps_id}.{img_type}'
tri_plot(xs, ys, losses, save_as, 'Validation loss', colour_map, clim_range)
"""

"""
# Accuracy plot

# Create accuracies plot
#save_as = f'analysis_accuracies_id-{static_hps_id}.{img_type}'
#accuracies = np.dot(accuracies, 100) # Turn into percentage
#colour_map = 'gist_rainbow_r'#'rainbow'#'jet'
#clim_range = [40, 100]
#clim_range = None
#plot_stuff(accuracies, save_as, 'Validation accuracy / %', colour_map, clim_range)
"""

plt.show()
