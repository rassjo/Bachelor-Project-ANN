import ANN as ann
import activation_functions as act
import synthetic_data_generation as sdg
import classification_statistics as cs
import rng_utils as ru
import matplotlib.pyplot as plt

# Define hyperparameters

# Use hyperparameters like these to check small scale step-by-step stuff
hps = {'lrn_rate': 0.1,
      'epochs': 1000,
      'val_mul': 4,
      'hidden': 15,
      'l2': 0.0,
      'dataset': '10d_intercept',
      'dropout': 1.0,
      'is_janky': True}
print_debugging_text = False

# Define data seed and ann seed
data_seed = ann_seed = 1
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

#Properties of all the layers
# Recipe for defining a layer: [number of nodes, activation function, L2, dropout]
layer_defines = [[hps['hidden'], act.tanh, hps['l2'], hps['dropout']],
                [1, act.sig, hps['l2'], hps['dropout']]] # dropout applies to the inputs
print("\ninitialising model...") if not print_debugging_text else None
ann_model = ann.Model(input_dim, layer_defines, ann_rng, is_debugging=print_debugging_text, is_janky=hps['is_janky'])
print("finished intialising model!") if not print_debugging_text else None

print("\ntraining...") if not print_debugging_text else None
ann_model.train(trn, val, hps['lrn_rate'], hps['epochs'], should_save_intermediary_history=True) #training, validation, lrn_rate, epochs, minibatchsize=0
print("finished training!") if not print_debugging_text else None

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