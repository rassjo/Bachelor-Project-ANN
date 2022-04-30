import hyperparameter_search as hs
import sys

# Define static and variable hyperparameters
static_hps = {'lrn_rate': 0.1,
      'epochs': None,
      'val_mul': 4,
      'hidden': 20,
      'patterns': None, # This is for overriding the number of training patterns belonging to each distribution. # i.e. 5 --> 5:5
      'dataset': 'hard_10d'}
variable_hps = [hs.variable_hp('dropout', [0, 1], is_log_dist=False, is_rev_open=True),
      hs.variable_hp('l2', [1e-6, 1], is_log_dist=True, is_rev_open=False)]

epochs = [1000, 4000]

#override_patterns = [5, 8, 15, 28, 50, 88, 158, 281, 500]
override_patterns = [5, 50, 500] #[50]

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
should_make_plots = False
img_type = 'pdf' # Change to .pdf if generating images for thesis
max_iterations = 8 # This is the max number of iterations to search before stopping

for epoch in epochs:
    static_hps['epochs'] = epoch
    
    for patterns in override_patterns:
        search_seed = temp_seed * patterns # just to jumble things up
    
        # Set the number of patterns
        static_hps['patterns'] = patterns
    
        # Perform dropout & l2 dual random search
        hs.dual_hyperparameter_search(static_hps, variable_hps, data_seed, ann_seed, search_seed, max_iterations, should_make_plots, img_type)