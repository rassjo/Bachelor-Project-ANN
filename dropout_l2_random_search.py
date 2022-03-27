import hyperparameter_search as hs
import sys

# Define static and variable hyperparameters
static_hps = {'lrn_rate': 0.1,
      'epochs': 1000,
      'val_mul': 4,
      'hidden': 15,
      'patterns': None,
      'dataset': '2d_intercept'}
variable_hps = [hs.variable_hp('dropout', [0.5, 1], is_log_dist=False, is_rev_open=True),
      hs.variable_hp('l2', [1e-6, 1], is_log_dist=True, is_rev_open=False)]

# Define data seed and ann seed
temp_seed = 1
data_seed = ann_seed = temp_seed

# Define search seed
try:
    temp_seed = int(sys.argv[1])
except:
    temp_seed = 1
search_seed = temp_seed

# Declare variables for the random search
should_make_plots = True
img_type = 'pdf' # Change to .pdf if generating images for thesis
max_iterations = 64 # This is the of variations to search

# Perform dropout & l2 dual random search
hs.dual_hyperparameter_search(static_hps, variable_hps, data_seed, ann_seed, search_seed, max_iterations, should_make_plots, img_type)
