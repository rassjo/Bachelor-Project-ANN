import numpy as np
import activation_functions as act
from synthetic_data_generation import *

#Remember:
#Weights on the same ROW act on the same node
#Weights on the same COLUMN come from the same node

class Model:
    def __init__(self, input_dim, layer_defines, rng=np.random.default_rng()):
        self.layers = [] #This is where we'll put all the layers we make
        columns = input_dim #Number of columns to use for the first weight matrix
        #Make as many layers as we have defined properties for
        for layer_properties in layer_defines:
            #Create a dense layer
            self.layers.append(Layer_Dense(columns,*layer_properties))
            #Then we want to know how many columns the weight matrix in the
            #next layer should have, by looking at how many rows (i.e. how the
            #many nodes) the current one has.
            columns = self.layers[-1].weights.shape[0]
    
    def feed_forward(self, X):
        for layer in self.layers:
            layer.input = X
            next_layer_input = layer.calc_output(X)
            X = next_layer_input
        
    
    def train(self, X, target):
        #Right now this function only does online updating (one pattern)
        #Obviously this has to be changed, I just threw something together
        #quickly so I could test the backpropagation function.
        self.feed_forward(X)
        all_updates = self.backpropagate(self.layers[-1].output,target)
        self.layers.reverse()
        for i in range(0, len(self.layers)):
            self.layers[i].weights -= all_updates[i]
        self.layers.reverse()
            
    
    def backpropagate(self,y,d):
        num_layers = len(self.layers) #Count the layers
        self.layers.reverse() #Reverse the order of the layers so we can do BACKpropagating
        #Prepare a place to save all the updates
        all_updates = []
        #----------------------------------------------------------------------
        #The code below is based on the equations on page 25 in the FYTN14
        #lecture notes.
        #----------------------------------------------------------------------
        #Set the first round of deltas to the hard-coded loss term that is
        #obtained when picking sigmoid output and binary cross-entropy loss.
        #NOTE: We might want to change this later on in case we want to
        #expand the scope of problems our network to solve, however, right
        #now we just need to be able to train it so I'll leave this for now.
        deltas = loss(y,d)
        #Now we want to calculate the update for all weights, layer by layer
        for i in range(0, num_layers):
            #For the current layer we need to know the weights
            #Previous layer refers to the layer that comes before in the
            #feed-forward step
            current = self.layers[i].weights
            prev_output = self.layers[i].input
            #Make an update matrix for the current layer
            update = np.zeros(current.shape)
            #Equation 2.11 in FYTN14 Lecture Notes
            for row in range(0,current.shape[0]):
                update[row] = deltas[row]*prev_output
            #Save the updates for the current layers
            all_updates.append(update)
            #If we haven't reached the final layer in the backpropagation
            #procedure, we need to calculate the deltas for the next step
            if i+1 != num_layers:
                new_deltas = []
                #This loop computes equation 2.10 in FYTN14 Lecture Notes
                for j in range(0, current.shape[1]):
                    delta_sum = sum(deltas*current.T[j])
                    derivative = self.layers[i+1].activation['der'](prev_output[j])
                    new_deltas.append(delta_sum*derivative)
                #Replace the deltas with the new values
                deltas = new_deltas
        #When we are done return the list of all layers to it's original state
        self.layers.reverse()
        print(all_updates)
        return all_updates
        

class Layer_Dense:   
    #Initialize the dense layer with inputs random weights & biases and
    #the right activation function
    def __init__(self, dim, nodes, activation, rng=np.random.default_rng()):
        self.weights = rng.standard_normal(size = (nodes, dim))
        self.biases = rng.standard_normal(size = (1, nodes))    
        self.activation = activation
        self.input = None
        self.output = None
    
    #Calculate the output of the layer
    def calc_output(self, X):
        self.input = X
        argument = (np.dot(self.weights, self.input) + self.biases).flatten()
        self.output = self.activation['act'](argument)
        #We want to flatten the output to turn it into a 1D array, otherwise
        #it will be a 2D array which causes problems when we want to check
        #the input dimension to set up the next layer.
        return self.output

def Error(finaloutput,targets):
    y = finaloutput
    d = targets
    N = len(targets)
    E = -1/N*(np.dot(d,np.log(y))+np.dot((1-d),np.log(1-y)))
    return E

def classification_loss(y,d):
    return y-d

loss = np.vectorize(classification_loss)

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
trn, val = generate_datasets('simple', try_plot=False)    
x_trn, d_trn = trn[0], trn[1]
x_val, d_val = val[0], val[1] 
#-----------------------------------------------------------------------------

#The input to feed into the first layer
initialInput = x_trn[0] # For now, just use the first generated training input

input_dim = len(initialInput)
#Properties of all the layers
#Recipe for defining a layer: [number of nodes, activation function]
layer_defines = [[1, act.tanh],
                 [1, act.sig]]

#Create the model based on the above
test = Model(input_dim, layer_defines, ann_rng)

test.feed_forward(initialInput)

def check_results(model):
    i=1
    for layer in model.layers:
    
        print(f"\nWeights of layer {i}: \n {layer.weights}")
        print(f"\nBiases of layer {i}: \n {layer.biases}")
        print(f"\nOutput of  layer {i}: \n {layer.output}")
        i += 1
    return layer.output

target = 0

#Check results
answer1 = check_results(test)

#Train the model (right now on a single pattern)
test.train(initialInput, target) #Try both target 0 and 1

test.feed_forward(initialInput)

#Check results again
answer2 = check_results(test)

#Did it improve?
print(f"Distance from target before training: {abs(target - answer1)}")
print(f"Distance from target after training: {abs(target - answer2)}")
