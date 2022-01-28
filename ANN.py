import numpy as np
from synthetic_data_generation import *

#Remember:
#Weights on the same ROW act on the same node
#Weights on the same COLUMN come from the same node

class Model:
    def __init__(self, initial, defines, rng=np.random.default_rng()):
        self.layers = [] #This is where we'll put all the layers we make
        input_data = initial #Input data to use for the first layer
        #Make as many layers as we have defined properties for
        for properties in defines:
            #Create a dense layer
            self.layers.append(Layer_Dense(input_data,*properties))
            #Then we want to feed the output of this layer to the next one
            input_data = self.layers[-1].output()


class Layer_Dense:   
    #Initialize the dense layer with inputs random weights & biases and
    #the right activation function
    def __init__(self, X, nodes, activation, rng=np.random.default_rng()):
        #Dimension of the input to the layer, needed to get the
        #right size on the weight matrix
        dim = len(X)
        self.weights = rng.standard_normal(size = (nodes, dim))
        self.biases = rng.standard_normal(size = (1, nodes))    
        self.input = X
        self.activation = activation
    
    #Calculate the output of the layer
    def output(self):
        argument = np.dot(self.weights, self.input) + self.biases
        return self.activation(argument).flatten()
        #We want to flatten the output to turn it into a 1D array, otherwise
        #it will be a 2D array which causes problems when we want to check
        #the input dimension to set up the next layer.

def Error(finaloutput,targets):
    y = finaloutput
    d = targets
    N = len(targets)
    E= -1/N*(np.dot(d,np.log(y))+np.dot((1-d),np.log(1-y)))
    return E

#------------------------------------------------------------------------------
# Create random number generators:
# seed == -1 for random rng, seed >= 0 for fixed rng (seed should be integer)
data_seed = -1
ann_seed = data_seed

def generate_rng(seed):
    # seed should be an integer
    # -1 is for random rng, integer >= 0 for fixed
    if (seed != -1):
        return np.random.default_rng(data_seed)
    return np.random.default_rng()

data_rng = generate_rng(data_seed)
ann_rng = generate_rng(ann_seed)

#-----------------------------------------------------------------------------
# Import data
trn, val = generate_datasets('smiley', try_plot=True)    
x_trn, d_trn = trn[0], trn[1]
x_val, d_val = val[0], val[1] 

#------------------------------------------------------------------------------
#Activation function definitions:

linear = lambda y: y

act_linear = np.vectorize(linear)
#Vectorize allows the function to act elementwise on arrays

ReLU = lambda y: max(0.0,y)
#NOTE: If we use 0 instead of 0.0 numpy will return an int32 array when the
#first entry in the array is 0, meaning it rounds the other numbers down to
#the closest integer as well!

act_ReLU = np.vectorize(ReLU)

logistic = lambda y: 1/(1+np.e**(-y))

act_logistic = np.vectorize(logistic)

d_logistic = lambda y: y*(1.0 - y)

tanh = lambda y: np.tanh(y)

act_tanh = np.vectorize(tanh)

d_tanh = lambda y: 1.0 - y**2

#------------------------------------------------------------------------------
#The input to feed into the first layer
initialInput = x_trn[0] # For now, just use the first generated training input

#Properties of all the layers
#Recipe for defining a layer: [number of nodes, activation function]
layers = [[1, act_linear],
          [3, act_linear],
          [2, act_linear],
          [4, act_ReLU]]

#Create the model based on the above
model = Model(initialInput, layers, ann_rng)

#Check the results
i=1
for layer in model.layers:

    print(f"\nWeights of layer {i}: \n {layer.weights}")
    print(f"\nBiases of layer {i}: \n {layer.biases}")
    print(f"\nOutput of  layer {i}: \n {layer.output()}")
    i += 1


