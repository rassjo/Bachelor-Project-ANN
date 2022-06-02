import hyperparameter_search as hs
import sys

# 1) Collect the bulk of the points
print("Collecting bulk of points:")
# Define static and variable hyperparameters
static_hps = {'lrn_rate': 0.1,
      'epochs': None,
      'val_mul': 4,
      'hidden': 20,
      'dataset': 'hard_10d',
      'dropout': 1.0}
num_parts = 9
num_samples = 5
# Remember that patterns is the number of patterns belonging to each distribution!!!
variable_hps = [hs.variable_hp('patterns', [5, 500.5], is_random_dist=False, num_parts=num_parts, is_log_dist=True, is_rev_open=True, make_int=True), # consider using an ODD number of parts
      hs.variable_hp('l2', [1e-5, 1], is_log_dist=True, is_rev_open=False)]

epochs = [1000, 4000]

# Define data seed and ann seed
try:
    temp_seed = int(sys.argv[1])
except:
    temp_seed = 1
search_seed = data_seed = ann_seed = temp_seed
# Uncomment to jumble up the search_seed between dropout rates
#search_seed += 20

# Declare variables for the random search
should_make_plots = False
img_type = 'pdf' # Change to .pdf if generating images for thesis
max_iterations = num_parts * num_samples # This is the number of variations to search

# Perform dropout & l2 dual hybrid search
for epoch in epochs:
      static_hps['epochs'] = epoch
      hs.dual_hyperparameter_search(static_hps, variable_hps, data_seed, ann_seed, search_seed, max_iterations, should_make_plots, img_type)


# 2) Collect the zero-l2 points
print("\nCollecting zero-l2 points...")

max_iterations = num_parts

variable_hps = [hs.variable_hp('patterns', [5, 500.5], is_random_dist=False, num_parts=num_parts, is_log_dist=True, is_rev_open=True, make_int=True), # consider using an ODD number of parts
      hs.variable_hp('l2', [0, 0], is_log_dist=False, is_rev_open=False)]

for epoch in epochs:
    static_hps['epochs'] = epoch
    hs.dual_hyperparameter_search(static_hps, variable_hps, data_seed, ann_seed, search_seed, max_iterations, should_make_plots, img_type)
