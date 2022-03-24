import ANN as ann
import numpy as np
import activation_functions as act
import synthetic_data_generation as sdg
import matplotlib.pyplot as plt
import classification_statistics as cs
from id_generator import best_hash
import txt_utils as tu
import rng_utils as ru
import sys

"""
To do:

"""

# Define static and variable hyperparameters
static_hps = {'lrn_rate': 0.1,
      'epochs': 1000,
      'val_mul': 5,
      'hidden': 15,
      'dropout': 0.9,
      'dataset': '2d_engulfed'}

# Define data seed and ann seed
temp_seed = 1
data_seed = ann_seed = temp_seed

# Define search seed
try:
    temp_seed = int(sys.argv[1])
except:
    temp_seed = 1
search_seed = temp_seed

# Generate (hash) id from static hyperparameters
static_hps_str = str(list(static_hps.values()))
static_hps_id = best_hash(static_hps_str)

# Define the range of (training) patterns and l2-strengths to try
patterns_range = [10, 100]
l2_range = [1e-6, 1]

# Create local directories for organising text files and figures
results_dir = 'patterns_l2_random_search_results'
tu.create_local_dir(results_dir)
id_dir = f'id-{static_hps_id}'
tu.create_local_dir(f'{results_dir}/{id_dir}')
seed_dir = f'data-seed-{data_seed:02d}_ann-seed-{ann_seed:02d}_search-seed-{search_seed:02d}'
tu.create_local_dir(f'{results_dir}/{id_dir}/{seed_dir}')

# Create text file
txt_name = f'{results_dir}/{id_dir}/{seed_dir}/results.txt'
meta_data = [f'ID: {static_hps_id}\n', 
        f'Static hyperparameters: {static_hps}',
        f'Data seed: {data_seed:02d}',
        f'ANN seed: {ann_seed:02d}',
        f'Search seed: {search_seed:02d}',
        f'Training patterns range: [{patterns_range[0]}, {patterns_range[1]}]',
        f'L2 range: [{l2_range[0]}, {l2_range[1]})']
column_labels = 'training patterns ; l2 strength ; final validation loss ; final validation confusion matrix'
tu.write_hyperparameters(txt_name, meta_data, column_labels)

# If text file already includes datapoints, then read them
# -- for continuing crashed runs whilst maintaining deterministic search rng.
hyperparameters_to_stats = tu.unload_results(txt_name)
hyperparameters = []
if not isinstance(hyperparameters_to_stats, type(None)):
    hyperparameters = list(hyperparameters_to_stats.keys())

# Generate the search rng from a fixed seed
search_rng = ru.generate_rng(search_seed)

# Declare variables for the random search
should_make_plots = True
img_type = 'pdf' # Change to .pdf if generating images for thesis
i = 0 # Iteration counter
max_iterations = 64 # This is the of variations to search

while i < max_iterations: # Essentially while True... just a bit safer, in case I die and never quit the program -- but then it's not my problem anyway
    # Generate random dropout rate from the half-open interval (dropout_range[0], dropout_range[1]]
    # Creating a distribution that is uniform in log10
    if patterns_range[0] != patterns_range[1]:
        patterns_pow_range = np.log10(patterns_range)
        patterns_pow = search_rng.random() # Generate uniformly distributed random number from the half-open interval (0, 1]
        patterns_pow *= (patterns_pow_range[1] - patterns_pow_range[0]) # Reduce the range
        patterns_pow += patterns_pow_range[0] # Displace the range
        patterns = 10**patterns_pow # Use the random number to generate log uniform l2
    else:
        patterns = patterns_range[0]
    patterns = int(patterns) # Ensure patterns is an integer

    # Generate random l2 strength from the half-open interval [l2_range[0], l2_range[1])
    # Creating a distribution that is uniform in log10
    if l2_range[0] != l2_range[1]:
        l2_pow_range = np.log10(l2_range)
        l2_pow = search_rng.random() # Generate uniformly distributed random number from the half-open interval [0, 1)
        l2_pow *= (l2_pow_range[1] - l2_pow_range[0]) # Reduce the range
        l2_pow += l2_pow_range[0] # Displace the range
        l2 = 10**l2_pow # Use the random number to generate log uniform l2
    else:
        l2 = l2_range[0]

    # Skip pre-existing datapoints (for resuming crashed runs from where they left off)
    if (patterns, l2) in hyperparameters:
        print(f"These hyperparameters already exist, skipping (i={i})!")
        i += 1
        continue

    if (patterns_range[0] == patterns_range[1]) and (l2_range[0] == l2_range[1]):
        # If there is only a single possible choice of dropout and l2, then only run a single iteration.
        i = max_iterations

    # Regenerate the data-rng and ann-rng from the fixed seed
    data_rng = ru.generate_rng(data_seed)
    ann_rng = ru.generate_rng(ann_seed)

    # Import data
    trn, val = sdg.generate_datasets(static_hps['dataset'],
                                 val_mul=static_hps['val_mul'],
                                 override_patterns=patterns,
                                 try_plot=should_make_plots,
                                 rng=data_rng)

    # Get the input dimension from the training data
    input_dim = len(trn[0][0]) 

    # Properties of all the layers
    # Recipe for defining a layer: [number of nodes, activation function, L2, dropout]
    layer_defines = [[static_hps['hidden'], act.tanh, l2, static_hps['dropout']],
                     [1, act.sig, l2, static_hps['dropout']]]
    ann_model = ann.Model(input_dim, layer_defines, ann_rng)

    # Train the network
    ann_model.train(trn, val, static_hps['lrn_rate'], static_hps['epochs'], 0, history_plot=should_make_plots)

    # Get final validation loss
    fin_val_loss = ann_model.history['val'][-1] 

    # Seperate validation in patterns and targets
    val_patterns = val[0]
    val_targets = val[1]

    # Get final validation accuracy
    val_outputs = ann_model.feed_all_patterns(val_patterns) # Collect the outputs for all the patterns
    val_confusion_matrix = cs.construct_confusion_matrix(val_outputs, val_targets) # Construct confusion matrix

    # Append the final validation loss to the text file
    with open(txt_name, 'a') as f:
        f.write(f'$ {patterns} ; {l2} ; {fin_val_loss} ; {val_confusion_matrix.tolist()}\n')

    # Save the error over epochs plot
    if (should_make_plots):
        # Save the error over epochs plot
        plt.savefig(f'{results_dir}/{id_dir}/{seed_dir}/error-over-epochs_patterns-{patterns}_l2-{l2}.{img_type}') 

        # Plot the decision boundary
        cs.decision_boundary(val_patterns, val_targets, ann_model)
        plt.savefig(f'{results_dir}/{id_dir}/{seed_dir}/validation-decision-boundary_patterns-{patterns}_l2-{l2}.{img_type}')

    # Increment counter
    i += 1

print('Finished!')