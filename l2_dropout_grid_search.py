import ANN_w_dropout as ann
import numpy as np
import activation_functions as act
import synthetic_data_generation as sdg
import matplotlib.pyplot as plt
import classification_statistics as cs
from id_generator import best_hash
import sys

"""
To do:
1. Add random hyperparameter optimisation search method (as opposed to grid search).
2. Read about dropout and L2 together from the dropout paper, to figure out why it isn't working better in comparison for the optimal. OHHH maybe that's something, the optimal doesn't get improved!! Just non-optimal values... hmmm. why is dropout only better 50% of time...? is it?
"""

def write_hyperparameters(name,static_hps,identifier,seed):
    # If the document already exists, then do nothing.
    try:
        open(name, "x")
    except:
        raise Exception(f"Results from a run with these settings already exists! Please delete {name}!")
    
    # Overwrite the empty file with the hyperparameters
    with open(name, "w") as f:
        print(f"Static hyperparameters: {static_hps}\n", file=f)
        print(f"Seed: {seed:02d}\n", file=f)
        print(f"ID: {identifier}\n", file=f)
        print("Below are the results!\n", file=f)
        print("dropout rate ; l2 strength ; final validation loss\n", file=f)

def generate_rng(seed):
    # seed should be an integer
    # -1 is for random rng, integer >= 0 for fixed
    if (seed != -1):
        return np.random.default_rng(seed)
    else:
        raise Exception(f'Please set a fixed seed, otherwise results will not be reproducable!')

# Define static and variable hyperparameters
static_hps = {'lrn_rate': 0.1,
      'epochs': 500,
      'val_mul': 5,
      'hidden': 15,
      'dataset': 'lagom'}
variable_hps = {'dropout': 1, # The probability of KEEPing a node (applies to input nodes and middle nodes)
      'l2': 0,}

# Define seed
try:
    seed = int(sys.argv[1])
except:
    seed = 1
data_seed = ann_seed = seed

# Create text file
static_hps_str = str(list(static_hps.values()))
static_hps_id = best_hash(static_hps_str)
txt_name = f'patterns_lambda_id-{static_hps_id}_seed-{seed:02d}.txt'
write_hyperparameters(txt_name, static_hps, static_hps_id, seed)

# Define the lists of dropout rates and l2 strengths to try
dropouts_to_try = [1, 0.9, 0.8, 0.7]
l2s_to_try = [0, 0.001, 0.01, 0.1]

# Create a dictionary of {dropout: {l2: validation_loss}}
grid_search = {}

# Generate the data rng from a fixed seed
data_rng = generate_rng(data_seed)

# Import data
trn, val = sdg.generate_datasets(static_hps['dataset'],
                                 val_mul=static_hps['val_mul'],
                                 try_plot=True,
                                 rng=data_rng)

# Get the input dimension from the training data
input_dim = len(trn[0][0]) 

# Grid-search over dropout and l2 for optimal combination
for dropout in dropouts_to_try:
    # Vary dropout rate
    variable_hps['dropout'] = dropout

    # Prepare grid_search 'l2: loss' dictionary
    grid_search[dropout] = {}

    for l2 in l2s_to_try:
        # Vary L2 strength
        variable_hps['l2'] = l2

        # Regenerate the ann rng from the fixed seed
        ann_rng = generate_rng(ann_seed)

        # Properties of all the layers
        # Recipe for defining a layer: [number of nodes, activation function, L2, dropout]
        layer_defines = [[static_hps['hidden'], act.tanh, variable_hps['l2'], variable_hps['dropout']],
                        [1, act.sig, variable_hps['l2'], variable_hps['dropout']]]
        ann_model = ann.Model(input_dim, layer_defines, ann_rng)

        # Train the network
        ann_model.train(trn, val, static_hps['lrn_rate'], static_hps['epochs'], 0, history_plot=True)

        # Get final validation loss
        fin_val_loss = ann_model.history['val'][-1] 

        # Update dictionary
        grid_search[dropout][l2] = fin_val_loss

        # Append the final validation loss to the text file
        with open(txt_name, 'a') as f:
            f.write(f'$ {dropout} ;  {l2} ; {fin_val_loss}\n')

        # Save the error over epochs plot
        plt.savefig(f'error-over-epochs_dropout-{dropout}_l2-{l2}_seed-{seed}.png') # Change to .pdf when generating images for thesis

# Get the minimum fin_val_losses, and the corresponding dropout and lambda values
min_val_loss = np.inf
for dropout in dropouts_to_try:
    for l2 in l2s_to_try:
        curr_val_loss = grid_search[dropout][l2]
        if curr_val_loss < min_val_loss:
            min_val_loss = curr_val_loss
            best_dropout = dropout
            best_l2 = l2

# Display results
print(f'Minimum validation loss: {min_val_loss}.')
print(f'Best dropout rate: {best_dropout}.')
print(f'Best L2 strength: {best_l2}.')
print()
print(grid_search) # Like a heatmap??
plt.show()
        



                                    


        

