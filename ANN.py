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
        
    
    def train(self, training, lrn_rate, epochs):
        history = []
        N = len(training) #number of patterns
        for t in range(0, epochs):
            self.weight_updates = [] #Where we store the total weight update
            self.bias_updates = [] #Where we store the total bias update
            for i in range(0,len(self.layers)):
                #Get the shapes of all the layers so we can abuse numpy array
                #addition later
                self.weight_updates.append(np.zeros(self.layers[i].weights.shape))
                self.bias_updates.append(np.zeros(self.layers[i].biases.shape))
            #We want it in the reverse order because the all_updates we get from
            #the backpropagation will be in the revere order
            self.weight_updates.reverse() ; self.bias_updates.reverse()
            #Now get the update for each pattern
            for n in range(0,len(training[0])):
                self.feed_forward(training[0][n])
                #Adding updates for every pattern to the total weight updates
                all_w_updates, all_b_updates = self.backpropagate(self.layers[-1].output,training[1][n])
                for i in range(0, len(self.layers)):
                    self.weight_updates[i] += all_w_updates[i]
                    self.bias_updates[i] += all_b_updates[i]
            
            
            #This should be called once all patterns have been used in a minibatch
            #Reverse this because weight_updates is reversed
            self.layers.reverse()
            #Actually update the weights
            for i in range(0, len(self.layers)):
                self.layers[i].weights -= lrn_rate*self.weight_updates[i]/N
                self.layers[i].biases -= lrn_rate*self.bias_updates[i]/N
            self.layers.reverse() #Return to original order
            self.weight_updates = [-lrn_rate*item/N for item in self.weight_updates]
            self.bias_updates = [-lrn_rate*item/N for item in self.bias_updates]
        
    def backpropagate(self,y,d):
        num_layers = len(self.layers) #Count the layers
        self.layers.reverse() #Reverse the order of the layers so we can do BACKpropagating
        #Prepare a place to save all the updates
        all_w_updates = [] ; all_b_updates = []
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
            #Previous output refers to the layer that comes before in the
            #feed-forward step
            current = self.layers[i].weights
            prev_output = self.layers[i].input
            #Make an update matrix for the current layer
            w_update = np.zeros(current.shape)
            b_update = np.zeros(current.shape[0])
            #Equation 2.11 in FYTN14 Lecture Notes
            for row in range(0,current.shape[0]):
                w_update[row] = deltas[row]*prev_output
                b_update[row] = deltas[row]
            #Save the updates for the current layers
            all_w_updates.append(w_update) ; all_b_updates.append(b_update)
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
        return all_w_updates, all_b_updates
        

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

def Error(y,d):
    #E = -(np.dot(d,np.log(y))+np.dot((1-d),np.log(1-y)))[0] save for later
    if d == 0:
        return(-np.log(1-y))
    elif d == 1:
        return(-np.log(y))

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
trn, val = generate_datasets('circle_ception', try_plot=True)    
#-----------------------------------------------------------------------------

input_dim = len(trn[0][0]) #Get the input dimension from the training data

#Properties of all the layers
#Recipe for defining a layer: [number of nodes, activation function]
layer_defines = [[4, act.tanh],
                 [4, act.tanh],
                 [1, act.sig]]

#Create the model based on the above
test = Model(input_dim, layer_defines, ann_rng)

def check_results(model, show=False):
    loss = 0
    N = len(trn[0])
    for n  in range(0,N):
        model.feed_forward(trn[0][n])
        #print("Pattern", n ," in: ", trn[0][n])
        if show:
            print("Pattern ", n ," out: ", model.layers[-1].output)
            print("Pattern ", n ," targ: ", trn[1][n])
        loss+=Error(model.layers[-1].output, trn[1][n])
    return loss/N

def check_layers(model):
    weights = [layer.weights for layer in model.layers]
    biases = [layer.biases for layer in model.layers]
    print("Weights: ", weights)
    print("Biases: ", biases)

#Check results
answer1 = check_results(test)

#check_layers(test)

test.train(trn,0.009,50)

#check_layers(test)

#Check results again
answer2 = check_results(test)

print("Loss before training", answer1)
print("Loss after training", answer2)
