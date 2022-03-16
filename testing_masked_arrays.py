import numpy as np
import activation_functions as act

rng = np.random.default_rng(2)

nodes = 4 # number of nodes on this layer
dim = 2 # number of nodes on previous layer
weights = rng.standard_normal(size = (nodes, dim))
biases = rng.standard_normal(size = (1, nodes))
#biases = np.array([0, 0, 0, 0])
dropout_mask = rng.choice(a=[False, True], size=dim, p=[0.5, 0.5]) # True means drop value, False means keep value.
""" This might be a useful check to see if all elements are dropped, then use a new mask instead.
result = np.all(dropout_mask == True)
if (result): # Need to turn this into a loop
    print("all elements in mask were true, trying again.")
    dropout_mask = rng.choice(a=[False, True], size=dim, p=[0.5, 0.5]) # True means drop value, False means keep value.
"""

self_input = rng.standard_normal(size = (dim)) # same size as dim

argument = (np.dot(weights, self_input) + biases).flatten()

input_w_dropout = np.ma.MaskedArray(self_input, dropout_mask, fill_value=0) 
argument_w_dropout = (np.ma.dot(weights, input_w_dropout, strict=True) + biases).flatten()

print(self_input)
print(argument)
print(dropout_mask)
print(input_w_dropout)
print("bias:" + str(biases))
print(weights)
print(argument_w_dropout)
