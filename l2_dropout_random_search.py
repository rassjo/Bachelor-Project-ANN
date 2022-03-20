import ANN_w_dropout as ann
import numpy as np
import activation_functions as act
import synthetic_data_generation as sdg
import matplotlib.pyplot as plt
import classification_statistics as cs
from id_generator import best_hash
import sys
import os

"""
To do:
2. Read about dropout and L2 together from the dropout paper, to figure out why it isn't working better in comparison for the optimal. OHHH maybe that's something, the optimal doesn't get improved!! Just non-optimal values... hmmm. why is dropout only better 50% of time...? is it?
"""

def create_local_dir(new_local_dir):
    # Create a specified local directory if it doesn't already exist
    cwd = os.getcwd()
    dir = os.path.join(cwd, new_local_dir)
    if not os.path.exists(dir):
        os.mkdir(dir)

def write_hyperparameters(name,static_hps,identifier,seed):
    # If the document already exists, then do nothing
    try:
        open(name, "x")
    except:
        # There is a problem that the search rng resets between runs, meaning repeat data points in the case of crashes
        # However, (I believe) we wish to keep the search rng deterministic for statistical purposes
        #return
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

# Define data seed and ann seed
#try:
#    temp_seed = int(sys.argv[1])
#except:
#    temp_seed = 1
temp_seed = 1 # Temporarily fixed, whilst just focusing on a single seed
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
results_dir = 'l2_dropout_random_search_results'
create_local_dir(results_dir)

id_dir = f'id-{static_hps_id}'
create_local_dir(f'{results_dir}/{id_dir}')

seed_dir = f'search-seed-{search_seed:02d}'
create_local_dir(f'{results_dir}/{id_dir}/{seed_dir}')

# Create text file
txt_name = f'{results_dir}/{id_dir}/{seed_dir}/results.txt'
write_hyperparameters(txt_name, static_hps, static_hps_id, data_seed)

# Define the range of dropout rates and l2 strengths to try
dropout_range = [0.5, 1]
l2_range = [0, 1]

# Create a dictionary of {dropout: {l2: validation_loss}}
grid_search = {}

# Generate the data rng from a fixed seed
data_rng = generate_rng(data_seed)

# Generate the search rng from a fixed seed
search_rng = generate_rng(search_seed)

# Import data
trn, val = sdg.generate_datasets(static_hps['dataset'],
                                 val_mul=static_hps['val_mul'],
                                 try_plot=True,
                                 rng=data_rng)

# Get the input dimension from the training data
input_dim = len(trn[0][0]) 

# Random-search over dropout and l2 for optimal combination
i = 0 # Iteration counter
while i < 50: # Essentially while True... just a bit safer, in case I die and never quit the program -- but then it's not my problem anyway
    # Generate random dropout rate from the half-open interval (dropout_range[0], dropout_range[1]]
    dropout = search_rng.random() # Generate random number from the half-open interval [0, 1)
    dropout = 1 - dropout # Swap interval openness to (0, 1] to avoid dropping all nodes
    dropout *= (dropout_range[1] - dropout_range[0]) # Reduce the range
    dropout += dropout_range[0] # Displace the range
    variable_hps['dropout'] = dropout

    # Generate random l2 strength from the half-open interval [l2_range[0], l2_range[1])
    is_l2_in_range = False
    while is_l2_in_range == False: # This could go bad... Not really happy with the chosen distribution
        l2 = search_rng.lognormal(mean = -3, sigma = 3.5) # Generate a random number from a log-normal distribution
        is_l2_in_range = l2 > l2_range[0] and l2 < l2_range[1]
    l2 *= (l2_range[1] - l2_range[0]) # Reduce the range
    l2 += l2_range[0] # Displace the range
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

    # Append the final validation loss to the text file
    with open(txt_name, 'a') as f:
        f.write(f'$ {dropout} ; {l2} ; {fin_val_loss}\n')

    # Save the error over epochs plot
    plt.savefig(f'{results_dir}/{id_dir}/{seed_dir}/error-over-epochs_dropout-{dropout}_l2-{l2}.png') # Change to .pdf when generating images for thesis

    # Increment counter
    i += 1
