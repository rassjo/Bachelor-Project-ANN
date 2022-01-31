import numpy as np
import activation_functions as act
from synthetic_data_generation import *

#Remember:
#Weights on the same ROW act on the same node
#Weights on the same COLUMN come from the same node

class Model:
    def __init__(self, initial, defines, rng=np.random.default_rng()):
        self.layers = [] #This is where we'll put all the layers we make
        self.initial = initial
        input_data = self.initial #Input data to use for the first layer
        #Make as many layers as we have defined properties for
        for properties in defines:
            #Create a dense layer
            self.layers.append(Layer_Dense(input_data,*properties))
            #Then we want to feed the output of this layer to the next one
            input_data = self.layers[-1].output()
    
    def train(self, target):
        #Right now this function only does online updating (one pattern)
        #Obviously this has to be changed, I just threw something together
        #quickly so I could test the backpropagation function.
        all_updates = self.backpropagate(self.layers[-1].output(),target)
        self.layers.reverse()
        for i in range(0, len(self.layers)):
            self.layers[i].weights -= all_updates[i]
        self.layers.reverse()
            
    
    def backpropagate(self,y,d):
        #Count the layers
        num_layers = len(self.layers)
        #Reverse the order of the layers so we can do BACKpropagatin
        self.layers.reverse()
        #Add this to prevent index error later
        self.layers.append(self.initial)
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
            #Previous layer refer to the layer that comes before in the
            #feed-forward step
            current = self.layers[i].weights ; previous = self.layers[i+1]
            #Make an update matrix for the current layer
            update = np.zeros(current.shape)
            #If we have arrived at the final hidden layer in the
            #backpropagation step, use the input values.
            if i+1 == num_layers:
                prev_output = self.initial
            #Otherwise, calculate the output of the previous layer.
            else:
                prev_output = previous.activation['act'](previous.argument)
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
                    derivative = previous.activation['der'](previous.argument[j])
                    new_deltas.append(delta_sum*derivative)
                #Replace the deltas with the new values
                deltas = new_deltas
        #When we are done return the list of all layers to it's original state
        self.layers.pop() ; self.layers.reverse()
        
        return all_updates
        

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
        self.argument = (np.dot(self.weights, self.input) + self.biases).flatten()
        return self.activation['act'](self.argument)
        #We want to flatten the output to turn it into a 1D array, otherwise
        #it will be a 2D array which causes problems when we want to check
        #the input dimension to set up the next layer.

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

#Properties of all the layers
#Recipe for defining a layer: [number of nodes, activation function]
layer_defines = [[1, act.tanh],
                 [1, act.sig]]

#Create the model based on the above
test = Model(initialInput, layer_defines, ann_rng)

def check_results(model):
    i=1
    for layer in model.layers:
    
        print(f"\nWeights of layer {i}: \n {layer.weights}")
        print(f"\nBiases of layer {i}: \n {layer.biases}")
        print(f"\nOutput of  layer {i}: \n {layer.output()}")
        i += 1
    return layer.output()

target = 0

#Check results
answer1 = check_results(test)

#Train the model (right now on a single pattern)
test.train(target) #Try both target 0 and 1

#Check results again
answer2 = check_results(test)

#Did it improve?
print(f"Distance from target before training: {abs(target - answer1)}")
print(f"Distance from target after training: {abs(target - answer2)}")
