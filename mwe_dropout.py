import ANN_w_dropout as ann
import numpy as np
import activation_functions as act
import synthetic_data_generation as sdg
import classification_statistics as cs
from id_generator import best_hash

def generate_rng(seed):
    # seed should be an integer
    # -1 is for random rng, integer >= 0 for fixed
    if (seed != -1):
        return np.random.default_rng(data_seed)
    else:
        raise Exception(f'Please set a fixed seed, otherwise results will not be reproducable!')

hp = {'lrn_rate': 0.1,
      'epochs': 1,
      #'lambdas': lambda_x,
      'val_mul': 1,
      'hidden': 4,
      'dataset': 'baby'}
lambd = 0

data_seed = ann_seed = 5
data_rng = generate_rng(data_seed)
ann_rng = generate_rng(ann_seed)

# Import data
trn, val = sdg.generate_datasets(hp["dataset"],
                                extra = 0,
                                val_mul = hp["val_mul"],
                                try_plot = False,
                                rng = data_rng)

input_dim = len(trn[0][0]) #Get the input dimension from the training data

x_trn = trn[0]
d_trn = trn[1]
    
x_val = val[0]
d_val = val[1]

#Properties of all the layers
# Recipe for defining a layer: [number of nodes, activation function, L2, dropout]
layer_defines = [[hp["hidden"], act.tanh, lambd, 0.8],
                [1, act.sig, lambd, 0.5]] # dropout applies to the inputs
test = ann.Model(input_dim, layer_defines, ann_rng)

# 1. get dropout mask for a layer
# 2. feedforward: apply dropout mask of ea. layer to respective inputs (using) 
# 3. backpropagation: apply dropout mask of ea. layer to 
    
#outputs = test.feed_all_patterns(x_trn) # Collect the outputs for all the inputs
#outputs = test.feed_all_patterns(x_val) # Collect the outputs for all the inputs

test.train(trn, val, hp["lrn_rate"], hp["epochs"]) #training, validation, lrn_rate, epochs, minibatchsize=0

#outputs = test.feed_all_patterns(x_trn) # Collect the outputs for all the inputs
#outputs = test.feed_all_patterns(x_val) # Collect the outputs for all the inputs

#for layer in test.layers:
    #print(layer.dropout_mask)

#print(outputs)
    
