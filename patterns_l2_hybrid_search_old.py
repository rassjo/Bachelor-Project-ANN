import hyperparameter_search_old as hs
import sys

# 1) Collect the bulk of the points
print("Collecting bulk of points:")
# Define static and variable hyperparameters
static_hps = {'lrn_rate': 0.1,
      'epochs': 3000,#1000
      'val_mul': 4,
      'hidden': 20,
      'dataset': 'hard_10d',
      'old_ANN': True,
      'fixed_initialisation': True}
num_parts = 6
num_samples = 10
# Remember that patterns is the number of patterns belonging to each distribution!!!
variable_hps = [hs.variable_hp('patterns', [5, 500.5], is_random_dist=False, num_parts=num_parts, is_log_dist=True, is_rev_open=False, make_int=True), # consider using an ODD number of parts
      hs.variable_hp('l2', [1e-6, 1e-1], is_log_dist=True, is_rev_open=False)]

# Define data seed and ann seed
try:
    temp_seed = int(sys.argv[1])
except:
    temp_seed = 1
search_seed = data_seed = ann_seed = temp_seed
# Uncomment to jumble up the search_seed between dropout rates
#search_seed += 10

# Declare variables for the random search
should_make_plots = False
img_type = 'pdf' # Change to .pdf if generating images for thesis
max_iterations = num_parts * num_samples # This is the number of variations to search

# Perform dropout & l2 dual hybrid search
hs.dual_hyperparameter_search(static_hps, variable_hps, data_seed, ann_seed, search_seed, max_iterations, should_make_plots, img_type)

"""

# 2) Collect the zero-l2 points
print("\nCollecting zero-l2 points...")

max_iterations = num_parts

variable_hps = [hs.variable_hp('patterns', [5, 500.5], is_random_dist=False, num_parts=num_parts, is_log_dist=True, is_rev_open=False, make_int=True), # consider using an ODD number of parts
      hs.variable_hp('l2', [0, 0], is_log_dist=False, is_rev_open=False)]

hs.dual_hyperparameter_search(static_hps, variable_hps, data_seed, ann_seed, search_seed, max_iterations, should_make_plots, img_type)


# 3) Collect the x10 pattern points
print("\nCollecting x10 pattern points...")

max_iterations = num_samples

variable_hps = [hs.variable_hp('patterns', [5000, 5000], is_random_dist=False, num_parts=1, is_log_dist=False, is_rev_open=False), # consider using an ODD number of parts
      hs.variable_hp('l2', [1e-6, 1e-1], is_log_dist=True, is_rev_open=False)]

hs.dual_hyperparameter_search(static_hps, variable_hps, data_seed, ann_seed, search_seed, max_iterations, should_make_plots, img_type)


# 4) Collect the zero-l2, x10 pattern point
print("\nCollecting zero-l2, x10 pattern points...")

max_iterations = 1

variable_hps = [hs.variable_hp('patterns', [5000, 5000], is_random_dist=False, num_parts=1, is_log_dist=False, is_rev_open=False), # consider using an ODD number of parts
      hs.variable_hp('l2', [0, 0], is_log_dist=False, is_rev_open=False)]

hs.dual_hyperparameter_search(static_hps, variable_hps, data_seed, ann_seed, search_seed, max_iterations, should_make_plots, img_type)

"""