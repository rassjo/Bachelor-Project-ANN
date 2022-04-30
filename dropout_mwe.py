import ANN as ann
import activation_functions as act
import synthetic_data_generation as sdg
import classification_statistics as cs
import rng_utils as ru
import matplotlib.pyplot as plt

# Define hyperparameters

# Use hyperparameters like these to check small scale step-by-step stuff
hps = {'lrn_rate': 0.1, # Learning rate
      'epochs': 2, # Number of epochs
      'val_mul': 1, # Factor of validation patterns to training patterns
      'hidden': 3, # Number of hidden nodes (single-hidden-layer)
      'l2': 0.0, # L2 strength
      'dataset': 'baby', # The data-set (Check 'data_presets.txt' for options)
      'dropout': 0.5} # The keep-rate probability applied to each input and hidden node
print_debugging_text = True

# Use hyperparameters like these to check large scale stuff
"""
hps = {'lrn_rate': 0.1,
      'epochs': 2,
      'val_mul': 4,
      'hidden': 3,
      'l2': 0.001,
      'dataset': 'baby',
      'dropout': 1.0}
"""
"""
hps = {'lrn_rate': 0.1,
      'epochs': 3000,
      'val_mul': 4,
      'hidden': 20,
      'l2': 0.0,
      'dataset': 'hard_10d',
      'dropout': 1.0,
      'old_ANN': False}
"""

# Define data seed and ann seed
data_seed = ann_seed = 2
data_rng = ru.generate_rng(data_seed)
ann_rng = ru.generate_rng(ann_seed)

# Import data
print("generating data...")
trn, val = sdg.generate_datasets(hps['dataset'],
                                val_mul = hps['val_mul'],
                                try_plot = False,
                                rng = data_rng)
print("finished generating data!")

# Seperate the training and validation patterns from their targets
x_trn, d_trn = trn[0], trn[1]
x_val, d_val = val[0], val[1]

input_dim = len(x_trn[0]) #Get the input dimension from the training data

# Properties of all the layers
# Recipe for defining a layer: [number of nodes, activation function, L2, dropout]
layer_defines = [[hps['hidden'], act.tanh, hps['l2'], 1.0], # setting the inputs to have no dropout
                [1, act.sig, hps['l2'], hps['dropout']]]

# Initialise ANN
print("\ninitialising model...") if not print_debugging_text else None
ann_model = ann.Model(input_dim, layer_defines, ann_rng, is_debugging=print_debugging_text)
print("finished intialising model!") if not print_debugging_text else None

# Train ANN
print("\ntraining...") if not print_debugging_text else None
ann_model.train(trn, val, hps['lrn_rate'], hps['epochs'], should_save_intermediary_history=True)
print("finished training!") if not print_debugging_text else None

# Plot error over epochs
print("\nshowing history...")
ann_model.show_history(hps['epochs'])
print("finished showing history!")

plt.show()

"""
print("\nshowing decision boundary... (only applicable for 1d and 2d datasets)")
print("Note that debugging text is disabled whilst generating decision boundaries, due to the shear number of feed forwards required.")
ann_model.is_debugging = False
if input_dim == 2:
    cs.decision_boundary(x_val, d_val, ann_model)
elif input_dim == 1:
    cs.decision_boundary_1d(x_val, d_val, ann_model)
print("finished showing decision boundary!")

plt.show()
"""
