from decimal import Clamped
from ftplib import all_errors
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import numpy as np
import classification_statistics as cs
import txt_utils as tu
import os
import matplotlib.tri as tri
import matplotlib.ticker

#rcParams['text.latex.preamble'] = [r'\usepackage{sfmath}']

# Define path to results.txt


# dropout & l2:
#static_hps_id = '384147b38d9f'
#static_hps_id = '1d0ffeda4835' # extra hidden nodes
#static_hps_id = '294b8b77c209' # low learning rate
#static_hps_id = '18616fcaad7b' # 2d dataset
#static_hps_id = '31ae0c6e5cc3' # extra low learning rate
#static_hps_id = '3a1625ccef53'
#static_hps_id = 'f5699d60a53'

# 1000 epochs
#static_hps_id = '7e1c2c93b8d' # 5 patterns
#static_hps_id = '366f4c71d23c' # 50 patterns
#static_hps_id = '10c4427ed35e' # 500 patterns

# 4000 epochs
#static_hps_id = '2166798a9a93' # 5 patterns
#static_hps_id = '10906c144a79' # 50 patterns
#static_hps_id = '2a48f9403264' # 500 patterns


#static_hps_id = '7e1c2c93b8d' #5 1000
#static_hps_id = '366f4c71d23c' #50 1000
#static_hps_id = '10c4427ed35e' #500 1000

#static_hps_id = '2166798a9a93' #5 4000
#static_hps_id = '10906c144a79' #50 4000
static_hps_id = '2a48f9403264' #500 4000

var1_name = 'dropout'
var1_label = 'Dropout rate'
var2_name = 'l2'
var2_label = 'L2 strength'
clim_range = [0.20510999114294412, 3.422007943213292 / 4]
#clim_range = None
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
            #if "data-seed-03" in subdir:
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

var1s = np.array(var1s)
var1s = 1-var1s

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
#title = f'Random hyperparameter search of {var2_name.capitalize()} and {var1_name.capitalize()}\n{static_hps_str}'
#x_label = f'{var1_label}'
#y_label = f'{var2_label}'
#title = f'Random hyperparameter search of L$_2$-strength & drop-rate'
x_label = f'Drop-rate $P$'
y_label = f'L$_2$-strength $\lambda$'
img_type = 'svg' # Change to .pdf if generating images for thesis

# Plot the scatter plot with a color bar
def plot_stuff(xs, ys, vals, save_as, colour_bar_label, colour_map, clim_range, x_lim, y_lim, is_x_log, is_y_log):
    fig, ax = plt.subplots()
    im = ax.scatter(xs, ys, marker=marker, c=vals, cmap=colour_map)
    
    if isinstance(clim_range, type(None)):
        im.set_clim(min(vals), max(vals)) # This is great, but lacks consistency between plots
    else:
        im.set_clim(clim_range[0], clim_range[1]) # Consistent, for multiple plots

    """ # Uncomment to include colour-bar
    cb = fig.colorbar(im, ax=ax, label=colour_bar_label)#, ticks=[-1, 0, 1])
    dumb_guess = 0.693
    custom_cb_ticks = list(cb.ax.get_yticks()) + [dumb_guess]
    cb.ax.set_yticks(custom_cb_ticks)
    cb.ax.set_ylim(min(vals), max(vals))
    """

    #ax.set_title('Raw runs')
    ax.set_xlabel(x_label)

    if (is_x_log):
        ax.set_xscale('log')
        ax.set_xlabel(x_label)
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

    ax.get_xaxis().set_visible(False)

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

    plt.savefig(f'{results_dir}/{id_dir}/{save_as}', bbox_inches='tight')


def tri_plot(xs, ys, xs_log, ys_log, vals, save_as, colour_bar_label, colour_map, clim_range):
    xs = np.array(xs)
    ys = np.array(ys)

    triang = tri.Triangulation(xs_log, ys_log)
    #triang = tri.Triangulation(xs, ys)
    #triang = tri.Triangulation(xs_ys[:, 0], xs_ys[:, 1])

    #refiner = tri.UniformTriRefiner(triang)
    #tri.TriAnalyzer(triang)

    xi, yi = np.meshgrid(np.linspace(min(xs_log), max(xs_log), 1000), np.linspace(min(ys_log), max(ys_log), 1000))
    #xi, yi = np.meshgrid(np.linspace(min(xs), max(xs), 1000), np.linspace(min(ys), max(ys), 1000))
    interp_cubic_geom = tri.CubicTriInterpolator(triang, vals, kind='min_E')
    zi_cubic_geom = interp_cubic_geom(xi, yi)

    for i in range(0, len(zi_cubic_geom)):
        for j in range(0, len(zi_cubic_geom[i])):
            if zi_cubic_geom[i,j] < 1.05*min(vals):
                zi_cubic_geom[i,j] = 1.05*min(vals)
            elif zi_cubic_geom[i,j] > 0.95*max(vals):
                zi_cubic_geom[i,j] = 0.95*max(vals)

    fig, ax = plt.subplots()
    #ax.set_aspect('equal')

    #zi_cubic_geom
    
    #ax.set_title('Interpolated runs')

    im1 = ax.imshow(zi_cubic_geom, cmap=colour_map, origin='lower', extent=[xs.min(), xs.max(), np.log10(ys).min(), np.log10(ys).max()], aspect='auto')

    if isinstance(clim_range, type(None)):
        im1.set_clim(min(vals), max(vals)) # This is great, but lacks consistency between plots
    else:
        im1.set_clim(clim_range[0], clim_range[1]) # Consistent, for multiple plots

    # Uncomment to include colour-bar
    cb = fig.colorbar(im1, ax=ax, label=colour_bar_label)#, ticks=[-1, 0, 1])
    custom_cb_ticklabels = ['{:.1f}'.format(a) for a in list(cb.ax.get_yticks())]
    custom_cb_ticklabels.remove('0.7')
    #custom_cb_ticklabels = ['{:.1f}'.format(a).rstrip('0').rstrip('.') for a in list(cb.ax.get_yticks())]
    dumb_guess = 0.693
    custom_cb_ticklabels.append(f'$\mathbf{{{dumb_guess:.3f}}}$')
    custom_cb_ticklabels.append(f'$\mathbf{{{clim_range[1]:.2f}+}}$')
    custom_cb_ticklabels.append(f'0.2')
    custom_cb_ticks = list(cb.ax.get_yticks()) + [dumb_guess, clim_range[1], clim_range[0]]
    custom_cb_ticks.remove(0.7)
    cb.ax.set_yticks(custom_cb_ticks)
    cb.ax.set_yticklabels(custom_cb_ticklabels)
    if isinstance(clim_range, type(None)):
        cb.ax.set_ylim(min(vals), max(vals))
    else:
        cb.ax.set_ylim(clim_range[0], clim_range[1])
     


    #ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    #ax.set_ylabel(f'log$_{{{10}}}$({y_label})')

    #ax.set_yscale('log') # kmt

    if not isinstance(x_lim, type(None)):
        ax.set_xlim(x_lim[0], x_lim[1])
    if not isinstance(y_lim, type(None)):
        ax.set_ylim(np.log10(y_lim[0]), np.log10(y_lim[1]))

    custom_ticks = ['$10^{{{:.0f}}}$'.format(a) for a in list(ax.get_yticks())]
    print(custom_ticks)

    ax.set_yticks(list(ax.get_yticks()))
    ax.set_yticklabels(custom_ticks)

    ticks = np.array([-(i*1/8) for i in range(1, 9)])
    ticks = -10**ticks
    final_ticks = np.array(ticks)
    for i in range(0, 5):
        new_ticks = ticks-1
        final_ticks = np.append(final_ticks, new_ticks)
        ticks = new_ticks
    final_ticks += 0.05
    print(final_ticks)
    ax.set_yticks(final_ticks, minor=True)


    #

    #y_minor = matplotlib.ticker.LogLocator(base = 10.0, numticks = 5)

    #ax.get_yaxis().set_visible(False)
    #ax.get_xaxis().set_visible(False)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')    
    plt.savefig(f'{results_dir}/{id_dir}/{save_as}', bbox_inches='tight')

    
# Create loss plot
save_as = f'analysis_loss_id-{static_hps_id}.{img_type}'
colour_map = 'gist_rainbow'

if clim_range != None:
    if min(losses) < clim_range[0]:
        print(f"Warning: Minimum loss {min(losses)} is below clim range.")
    if max(losses) > clim_range[1]:
        print(f"Warning: Maximum loss {max(losses)} is above clim range.")

print('min loss:', min(losses))
print('max loss:', max (losses))

plot_stuff(var1s, var2s, losses, save_as, 'Validation loss $E$', colour_map, clim_range, x_lim, y_lim, is_x_log, is_y_log)


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
tri_plot(var1s, var2s, xs, ys, losses, save_as, 'Validation loss $E$', colour_map, clim_range)


"""
# Accuracy plot

# Create accuracies plot
save_as = f'analysis_accuracies_id-{static_hps_id}.{img_type}'
accuracies = np.dot(accuracies, 100) # Turn into percentage
colour_map = f'{colour_map}_r' # Reverse the colour-map for accuracies
plot_stuff(var1s, var2s, accuracies, save_as, 'Validation accuracy / %', colour_map, clim_range, x_lim, y_lim, is_x_log, is_y_log)


# Shaded accuracies plot

# Standardise data
xs = var1s
xs_norm = np.linalg.norm(xs)
xs = var1s/xs_norm

ys = var2s
ys = np.log10(ys)
ys_norm = np.linalg.norm(ys)
ys = ys/ys_norm

save_as = f'tri_analysis_accuracies_id-{static_hps_id}.{img_type}'
tri_plot(var1s, var2s, xs, ys, accuracies, save_as, 'Validation accuracy / %', colour_map, clim_range)
"""



plt.show()