import matplotlib.pyplot as plt
import numpy as np
import os

# Random-search
search_seed = 1
id = '2794f615569a'
txt_name = f'l2_dropout_random_search_results/id-{id}/search-seed-{search_seed:02d}/results.txt'
is_log_plot = True

# Get all the results.txt files from a given id
cwd = os.getcwd()
rootdir = os.path.join(cwd, 'l2_dropout_random_search_results', f'id-{id}')

dropout_l2_to_loss = {}
def unload_results(txt_name):
    with open(txt_name, 'r') as data:
        for line in data:
            if line[0] != '$':
                continue
            values = line.split(';')

            dropout = float(values[0][2:-1])
            l2 = float(values[1][1:-1])
            loss = float(values[2][1:-1])

            dropout_l2_to_loss[(dropout, l2)] = loss

# Walk through the directories and subdirectories of rootdir to find the results.txt files
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if (file == 'results.txt'):
            # Open the file 
            txt_name = os.path.join(subdir, file)
            unload_results(txt_name)

# Grid-search
#data_seed = 1
#id = '89bd58040b0'
#txt_name = f'l2_dropout_grid_search_results/data-seed-{data_seed:02d}/patterns_lambda_id-{id}.txt'
#is_log_plot = True

# Seperate the dropouts, l2s and losses for intuitive plotting 
dropouts = []
l2s = []
losses = []
for key in dropout_l2_to_loss:
    dropouts.append(key[0])
    l2s.append(key[1])
    losses.append(dropout_l2_to_loss[key])

# Define the x, y and colormap values
xs = dropouts
ys = l2s
vals = losses

# Plot the scatter plot with a color bar
fig, ax = plt.subplots()
im = ax.scatter(xs, ys, marker='x', c=vals, cmap='gist_rainbow')#cmap='plasma')
im.set_clim(min(vals), max(vals))
fig.colorbar(im, ax=ax, label='Validation loss')
ax.set_title('Random hyperparameter search of L2 and dropout')
ax.set_xlabel('Dropout rate')
ax.set_xlim(0.5, 1)
if (is_log_plot):
    #ax.set_ylim(10e-5, 1)
    ax.set_yscale('log')
    ax.set_ylim(None, 1)
    ax.set_ylabel('log$_{10}$(L2 strength)')
else:
    ax.set_ylim(0, 1)
    ax.set_ylabel('L2 strength')
plt.savefig(f'l2_dropout_random_search_results/id-{id}/analysis.pdf')
plt.show()

