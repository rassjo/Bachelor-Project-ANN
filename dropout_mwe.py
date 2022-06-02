from xml.dom import VALIDATION_ERR
import ANN as ann
import activation_functions as act
import synthetic_data_generation as sdg
import classification_statistics as cs
import rng_utils as ru
import matplotlib.pyplot as plt

# Define hyperparameters

# Use hyperparameters like these to check small scale step-by-step stuff
"""
hps = {'lrn_rate': 0.1, # Learning rate
      'epochs': 2, # Number of epochs
      'val_mul': 1, # Factor of validation patterns to training patterns
      'hidden': 3, # Number of hidden nodes (single-hidden-layer)
      'l2': 0.0, # L2 strength
      'dataset': 'baby', # The data-set (Check 'data_presets.txt' for options)
      'dropout': 0.5} # The keep-rate probability applied to each input and hidden node
print_debugging_text = True
"""

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
hps = {'lrn_rate': 0.1,
      'epochs': 4000,
      'val_mul': 4,
      'hidden': 20,
      'l2': 0.0,
      'dataset': 'hard_10d',
      'patterns': None,
      'dropout': 1.0}
print_debugging_text = False

override_patterns = [100/2, 178/2, 316/2, 562/2, 1000/2]
override_patterns = [int(pattern) for pattern in override_patterns]
print(override_patterns)

ann_models = []

for patterns in override_patterns:
    hps['patterns'] = patterns

    # Define data seed and ann seed
    data_seed = ann_seed = 1
    data_rng = ru.generate_rng(data_seed)
    ann_rng = ru.generate_rng(ann_seed)

    # Import data
    print("generating data...")
    trn, val = sdg.generate_datasets(hps['dataset'],
                                    val_mul = hps['val_mul'],
                                    override_patterns= hps['patterns'],
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

    ann_models.append(ann_model)

    # Plot error over epochs
    print("\nshowing history...")
    ann_model.show_history(hps['epochs'])
    print("finished showing history!")

    save_as = f'error-over-epochs_patterns-{patterns}'
    plt.savefig(f'{save_as}.pdf')

import numpy as np
# Plot the error over all epochs
plt.figure()
plt.plot(np.arange(0, hps['epochs']+1), ann_models[0].history['val'], 'tab:blue', label='$N = 10^2$')
plt.plot(np.argmin(np.array(ann_models[0].history['val'])), np.min(np.array(ann_models[0].history['val'])), 'o', color = 'tab:blue')
plt.plot(np.arange(0, hps['epochs']+1), ann_models[1].history['val'], 'tab:orange', label='$N = 10^{{{2.25}}}$')
plt.plot(np.argmin(np.array(ann_models[1].history['val'])), np.min(np.array(ann_models[1].history['val'])), 'o', color = 'tab:orange')
plt.plot(np.arange(0, hps['epochs']+1), ann_models[2].history['val'], 'tab:green', label='$N = 10^{{{2.5}}}$')
plt.plot(np.argmin(np.array(ann_models[2].history['val'])), np.min(np.array(ann_models[2].history['val'])), 'o', color='tab:green')
plt.plot(np.arange(0, hps['epochs']+1), ann_models[3].history['val'], 'tab:red', label='$N = 10^{{{2.75}}}$')
plt.plot(np.argmin(np.array(ann_models[3].history['val'])), np.min(np.array(ann_models[3].history['val'])), 'o', color='tab:red')
plt.plot(np.arange(0, hps['epochs']+1), ann_models[4].history['val'], 'tab:purple', label='$N = 10^3$')
plt.plot(np.argmin(np.array(ann_models[4].history['val'])), np.min(np.array(ann_models[4].history['val'])), 'o', color='tab:purple')
plt.plot([1000, 1000], [0, 1.0], '--', color='k', linewidth=0.8)
#plt.plot([1000, 1000], [0, self.history['val'][1000+1]], '--k')
#plt.plot([1000], [self.history['val'][1000+1]], 'ok')
plt.xlabel('Epochs')
plt.ylabel('Validation loss $E$')
plt.xlim(0, hps['epochs'])
plt.ylim(0, 0.85)
#plt.title('Error over epochs')
plt.legend(loc=1, title='Training patterns')
plt.draw()

save_as = f'error-over-epochs_all-patterns'
plt.savefig(f'{save_as}.pdf')

# Plot the error over all epochs
plt.figure()
plt.plot(np.arange(0, hps['epochs']+1), ann_models[0].history['val'], 'tab:blue', label='$N = 10^2$')
plt.plot(np.argmin(np.array(ann_models[0].history['val'])), np.min(np.array(ann_models[0].history['val'])), 'o', color = 'tab:blue')
plt.plot(np.arange(0, hps['epochs']+1), ann_models[1].history['val'], 'tab:orange', label='$N = 10^{{{2.25}}}$')
plt.plot(np.argmin(np.array(ann_models[1].history['val'])), np.min(np.array(ann_models[1].history['val'])), 'o', color = 'tab:orange')
plt.plot(np.arange(0, hps['epochs']+1), ann_models[2].history['val'], 'tab:green', label='$N = 10^{{{2.5}}}$')
plt.plot(np.argmin(np.array(ann_models[2].history['val'])), np.min(np.array(ann_models[2].history['val'])), 'o', color='tab:green')
plt.plot(np.arange(0, hps['epochs']+1), ann_models[3].history['val'], 'tab:red', label='$N = 10^{{{2.75}}}$')
plt.plot(np.argmin(np.array(ann_models[3].history['val'])), np.min(np.array(ann_models[3].history['val'])), 'o', color='tab:red')
plt.plot(np.arange(0, hps['epochs']+1), ann_models[4].history['val'], 'tab:purple', label='$N = 10^3$')
plt.plot(np.argmin(np.array(ann_models[4].history['val'])), np.min(np.array(ann_models[4].history['val'])), 'o', color='tab:purple')
plt.plot(np.arange(0, hps['epochs']+1), ann_models[0].history['trn'], '--', color='tab:blue')
plt.plot(np.arange(0, hps['epochs']+1), ann_models[1].history['trn'], '--', color='tab:orange')
plt.plot(np.arange(0, hps['epochs']+1), ann_models[2].history['trn'], '--', color='tab:green')
plt.plot(np.arange(0, hps['epochs']+1), ann_models[3].history['trn'], '--', color='tab:red')
plt.plot(np.arange(0, hps['epochs']+1), ann_models[4].history['trn'], '--', color='tab:purple')
plt.plot([1000, 1000], [0, 1.0], '--', color='k', linewidth=0.8)
#plt.plot([1000, 1000], [0, self.history['val'][1000+1]], '--k')
#plt.plot([1000], [self.history['val'][1000+1]], 'ok')
plt.xlabel('Epochs')
plt.ylabel('Loss $E$')
plt.xlim(0, hps['epochs'])
plt.ylim(0, 0.82)
#plt.title('Error over epochs')
plt.legend(loc=1, title='Training patterns:')
plt.draw()

save_as = f'error-over-epochs_all-patterns_trn-included'
plt.savefig(f'{save_as}.pdf')

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
