from audioop import reverse
from unicodedata import name
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

class variable_hp:
    def __init__(self, name, range, log_dist = False, rev_open = False):
        self.name = name
        self.range = range
        self.log_dist = log_dist
        self.rev_open = rev_open
        self.val = None

    def random(self, search_rng=np.random.default_rng()):
        # Generate random value from the desired range, scaling and openness
        if self.range[0] != self.range[1]:
            if (self.log_dist):
                # Generate value from a log distribution
                val_pow_range = np.log10(self.range)
                val_pow = search_rng.random() # Generate uniformly distributed random number from the half-open interval [0, 1)
                if (self.rev_open):
                    val_pow = 1 - val_pow # Swap interval openness to (0, 1]
                val_pow *= (val_pow_range[1] - val_pow_range[0])
                val_pow += val_pow_range[0]
                val = 10**val_pow
            else:
                # Generate value from a linear distribution
                val = search_rng.random() # Generate random number from the half-open interval [0, 1)
                if (self.rev_open):
                    val = 1 - val # Swap interval openness to (0, 1]
                val *= (self.range[1] - self.range[0]) # Reduce the range
                val += self.range[0] # Displace the range
        else:
            # If the range only permits a single value, then just use that value
            val = self.range[0]
        self.val = val
        return self.val

# Define static and variable hyperparameters
static_hps = {'lrn_rate': 0.1,
      'epochs': 1000,
      'val_mul': 5,
      'hidden': 15,
      'dataset': '2d_engulfed'}
variable_hps = [variable_hp('dropout', [0.5, 1], log_dist=False, rev_open=True),
      variable_hp('l2', [1e-6, 1], log_dist=True, rev_open=False)]

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

# Create local directories for organising text files and figures
results_dir = f'{variable_hps[0].name}_{variable_hps[1].name}_random_search_results'
tu.create_local_dir(results_dir)
id_dir = f'id-{static_hps_id}'
tu.create_local_dir(f'{results_dir}/{id_dir}')
seed_dir = f'data-seed-{data_seed:02d}_ann-seed-{ann_seed:02d}_search-seed-{search_seed:02d}'
tu.create_local_dir(f'{results_dir}/{id_dir}/{seed_dir}')

# Create text file
txt_name = f'{results_dir}/{id_dir}/{seed_dir}/results.txt'
meta_data = [f'ID: {static_hps_id}', 
            f'Static hyperparameters: {static_hps}',
            f'Data seed: {data_seed:02d}',
            f'ANN seed: {ann_seed:02d}',
            f'Search seed: {search_seed:02d}'] 
for hp in variable_hps:
    meta_data.append(f'{hp.name} range: ' + ('(' if hp.rev_open else '[') + f'{hp.range[0]}, {hp.range[1]}' + (']' if hp.rev_open else ')'))
# Print the meta data
print('Intialising run with meta-data:')
for line in meta_data:
    print(line)
print('')
column_labels = 'dropout rate ; l2 strength ; final validation loss ; final validation confusion matrix'
tu.write_hyperparameters(txt_name, meta_data, column_labels)



# If text file already includes datapoints, then read them
# -- for continuing crashed runs whilst maintaining deterministic search rng.
old_hps_to_stats = tu.unload_results(txt_name)
old_hps = []
if not isinstance(old_hps_to_stats, type(None)):
    old_hps = list(old_hps_to_stats.keys())

# Generate the search rng from a fixed seed
search_rng = ru.generate_rng(search_seed)

# Generate the data rng from a fixed seed
data_rng = ru.generate_rng(data_seed)

# Declare variables for the random search
should_make_plots = True
img_type = 'pdf' # Change to .pdf if generating images for thesis
i = 0 # Iteration counter
max_iterations = 64 # This is the of variations to search

# Import data
trn, val = sdg.generate_datasets(static_hps['dataset'],
                                 val_mul=static_hps['val_mul'],
                                 try_plot=should_make_plots,
                                 rng=data_rng)

# Get the input dimension from the training data
input_dim = len(trn[0][0]) 

# Random-search over dropout and l2 for optimal combination
while i < max_iterations: # Essentially while True... just a bit safer, in case I die and never quit the program -- but then it's not my problem anyway
    hps = {}

    for key in static_hps:
        hps[key] = static_hps[key]
    
    dual_hps = ()
    for hp in variable_hps:
        hps[hp.name] = hp.random(search_rng)
        dual_hps = (*dual_hps, hp.val)

    # Skip pre-existing datapoints (for resuming crashed runs from where they left off)
    # Additionally, ends the run if only a single choice of dropout and l2.
    if dual_hps in old_hps:
        print(f"These hyperparameters already exist, skipping (i={i})!")
        i += 1
        continue

    # Regenerate the ann rng from the fixed seed
    ann_rng = ru.generate_rng(ann_seed)

    # Properties of all the layers
    # Recipe for defining a layer: [number of nodes, activation function, L2, dropout]
    layer_defines = [[hps['hidden'], act.tanh, hps['l2'], hps['dropout']],
                     [1, act.sig, hps['l2'], hps['dropout']]]
    ann_model = ann.Model(input_dim, layer_defines, ann_rng)

    # Train the network
    ann_model.train(trn, val, hps['lrn_rate'], hps['epochs'], 0, history_plot=should_make_plots)

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
        old_hps.append(dual_hps)
        f.write(f'$ {dual_hps[0]} ; {dual_hps[1]} ; {fin_val_loss} ; {val_confusion_matrix.tolist()}\n')

    # Save the error over epochs plot
    if (should_make_plots):
        plot_id = f'{variable_hps[0].name}-{dual_hps[0]}_{variable_hps[1].name}-{dual_hps[1]}'

        # Save the error over epochs plot
        plt.savefig(f'{results_dir}/{id_dir}/{seed_dir}/error-over-epochs_{plot_id}.{img_type}') 

        # Plot and save the decision boundary
        cs.decision_boundary(val_patterns, val_targets, ann_model)
        plt.savefig(f'{results_dir}/{id_dir}/{seed_dir}/validation-decision-boundary_{plot_id}.{img_type}')

    # Increment counter
    i += 1

plt.show()

print('Finished!')