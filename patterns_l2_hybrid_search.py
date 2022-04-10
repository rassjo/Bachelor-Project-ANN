import hyperparameter_search as hs
import sys

# Define static and variable hyperparameters
static_hps = {'lrn_rate': 0.05,
      'epochs': 1000,
      'val_mul': 4,
      'hidden': 20,
      'dataset': '10d_intercept',
      'dropout': 1.0}
variable_hps = [hs.variable_hp('patterns', [10, 1000], is_random_dist=False, num_parts=6, is_log_dist=True, is_rev_open=False, make_int=True),
      hs.variable_hp('l2', [1e-6, 1e-1], is_log_dist=True, is_rev_open=False)]

# Define data seed and ann seed
try:
    temp_seed = int(sys.argv[1])
except:
    temp_seed = 1
search_seed = data_seed = ann_seed = temp_seed

# Declare variables for the random search
should_make_plots = False
img_type = 'pdf' # Change to .pdf if generating images for thesis
max_iterations = 72 # This is the of variations to search

# Perform dropout & l2 dual hybrid search
hs.dual_hyperparameter_search(static_hps, variable_hps, data_seed, ann_seed, search_seed, max_iterations, should_make_plots, img_type)
