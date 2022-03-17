from logging import exception
import numpy as np
import activation_functions as act
import synthetic_data_generation as sdg
import classification_statistics as cs
import matplotlib.pyplot as plt
import sys
import warnings


debugging = False
#Remember:
#Weights on the same ROW act on the same node
#Weights on the same COLUMN come from the same node

# To do:
# 3. get the same optimal lambda plot, but with dropout 

class Model:
    def __init__(self, input_dim, layer_defines, rng=np.random.default_rng()):
        self.layers = [] #This is where we'll put all the layers we make
        columns = input_dim #Number of columns to use for the first weight matrix
        self.rng = rng
        #Make as many layers as we have defined properties for
        for layer_properties in layer_defines:
            #Create a dense layer
            self.layers.append(Layer_Dense(columns,*layer_properties,
                                           rng = self.rng))
            #Then we want to know how many columns the weight matrix in the
            #next layer should have, by looking at how many rows (i.e. how the
            #many nodes) the current one has.
            columns = self.layers[-1].weights.shape[0]

    def feed_forward(self, X):
        print("FEED FORWARD PATTERN:") if debugging else None
        for layer in self.layers:
            print("     NEW LAYER:") if debugging else None
            layer.input = X
            next_layer_input = layer.calc_output(X)
            print("          dropout mask: " + str(layer.dropout_mask)) if debugging else None
            print("          input w dropout: " + str(layer.get_input_w_dropout())) if debugging else None
            print("          bias: " + str(layer.biases)) if debugging else None
            print("          output: " + str(next_layer_input)) if debugging else None
            X = next_layer_input

    def feed_all_patterns(self, patterns):
        outputs = []
        for pattern in patterns:
            self.feed_forward(pattern) #Update the network's output...
            outputs.append(float(self.layers[-1].output)) #...and save it
        return np.array(outputs)

    def backpropagate(self, y, d):
        print("BACK PROPAGATE PATTERN:") if debugging else None
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
            layer = self.layers[i]
            print("     NEW LAYER:") if debugging else None

            #For the current layer we need to know the weights
            #Previous output refers to the layer that comes before in the
            #feed-forward step
            current = layer.weights
            prev_output = layer.get_input_w_dropout()
            print("          dropout mask: " + str(layer.dropout_mask)) if debugging else None
            print("          input w dropout: " + str(layer.get_input_w_dropout())) if debugging else None
            print("          bias: " + str(layer.biases)) if debugging else None

            #Make an update matrix for the current layer
            #w_update = np.zeros(current.shape)
            #b_update = np.zeros(current.shape[0])
            w_update = np.ma.zeros(current.shape)
            b_update = np.ma.zeros(current.shape[0])

            #Equation 2.11 in FYTN14 Lecture Notes
            for row in range(0,current.shape[0]):
                #w_update[row] = deltas[row]*prev_output
                w_update[row] = np.ma.dot(deltas[row], prev_output, strict=True)
                b_update[row] = deltas[row]
            print("          bias update: " + str(b_update)) if debugging else None
            print("          weight update: " + str(w_update)) if debugging else None

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

    def update_weights(self,training,minibatchsize,lrn_rate):
        #We want it in the reverse order because the all_updates we get from
        #the backpropagation will be in the reverse order
        self.weight_updates.reverse() ; self.bias_updates.reverse()

        #Now go through each pattern in the mini-batch
        for _pattern in range(0,minibatchsize):
            #Find the network's output for the pattern
            self.feed_forward(training[0][self.n])
            #Adding updates for every pattern to the total weight updates
            all_w_updates, all_b_updates = self.backpropagate(self.layers[-1].output,training[1][self.n])

            for i in range(0, len(self.layers)):
                #print("     NEW LAYER:")
                self.weight_updates[i] += all_w_updates[i]
                self.bias_updates[i] += all_b_updates[i]
            self.trn_output[self.n] = self.layers[-1].output[0]
            self.n += 1 #Increment the counter when we go to the next pattern

        #Now we have all of the weight updates for the current mini-batch!

        #Reverse the layers because weight_updates is reversed
        self.layers.reverse()
        #Actually update the weights
        for i in range(0, len(self.layers)):
          layer = self.layers[i] #Current layer
          #Update the weights with the result from backpropagation and
          #the derivative of the L2-term
          print("OLD:") if debugging else None
          print("weights: " + str(layer.weights)) if debugging else None
          print("biases: " + str(layer.biases)) if debugging else None
          layer.weights -= (lrn_rate*self.weight_updates[i]/minibatchsize + layer.l2_s*layer.weights)
          layer.biases -= lrn_rate*self.bias_updates[i]/minibatchsize
          print("NEW:") if debugging else None
          print("weights: " + str(layer.weights)) if debugging else None
          print("biases: " + str(layer.biases)) if debugging else None
        self.layers.reverse() #Return to original order

    def show_history(self,epochs):
        # Plot the error over all epochs
        plt.figure()
        plt.plot(np.arange(0,epochs+1), self.history['trn'], 'orange', label='Training error')
        plt.plot(np.arange(0,epochs+1), self.history['val'], 'blue', label='Validation error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Error over epochs')
        plt.legend()
        plt.show()

    def train(self, training, validation, lrn_rate, epochs, minibatchsize=0,
              save_val = False, history_plot = False):
        N = len(training[0]) #number of patterns in training data
        N_val = len(validation[0]) #number of patterns in validation data
        #If minibatchsize is 0 (i.e. default), do regular gradient descent
        if minibatchsize == 0:
            minibatchsize = N
        fixed_minibatchsize = minibatchsize #Save the true mini-batch size somewhere
        #Check if the mini-batch size is larger than the total amount of training patterns
        if minibatchsize > N:
            raise Exception(f'Your mini-batch size is too large! With the current training data it cannot exceed {N}.')

        #Number of extra patterns to be added onto the final mini-batch
        extra = N%fixed_minibatchsize
        #If the loss over epochs is supposed to be displayed and the user
        #forgot to also turn on saving validation loss after every epoch
        if history_plot:
            save_val = True
            
        self.history = {'trn':[],'val':[]} #This is where we will save the loss after each epoch

        # Temporarily turn off dropout (for loss stuff)
        for layer in self.layers:
            layer.disable_dropout()
            print("dropout_enabled: " + str(layer.dropout_enabled)) if debugging else None
        
        print("\nInitial training loss (dropout off)") if debugging else None
        #Get the initial training loss
        self.trn_output = self.feed_all_patterns(training[0])
        loss_array=ErrorV(np.array(self.trn_output),training[1])
        self.history['trn'].append(sum(loss_array)/N)

        print("\nInitial validation loss (dropout off)") if debugging else None
        #Get the initial validation loss
        self.val_output = self.feed_all_patterns(validation[0])
        loss_array=ErrorV(np.array(self.val_output),validation[1])
        self.history['val'].append(sum(loss_array)/N_val)

        #This loop is for going through the desired amount of epochs
        for epoch_nr in range(0, epochs):
            print(f"\nEpoch {epoch_nr + 1} (dropout on)") if debugging else None
            # Refresh the dropout mask for each layer.
            for layer in self.layers:
                layer.enable_dropout()
                print("dropout_enabled: " + str(layer.dropout_enabled)) if debugging else None
                layer.generate_dropout_mask(rng=self.rng)

            #This is where we save results after each training pattern
            self.trn_output = np.ma.zeros(N)
            self.n = 0 #This will keep track of which pattern we are at during the epoch
            #These lines randomize the pattern order for each new epoch
            p = np.random.permutation(len(training[0]))
            training = [training[0][p], training[1][p]]
            #Restore the mini-batch size to the original value
            minibatchsize = fixed_minibatchsize

            #Now it's time to go through all of the mini-batches
            for minibatch_nr in range(0,N//fixed_minibatchsize):
                #This makes sure the last minibatch gets any remaining patterns
                if minibatch_nr == N//fixed_minibatchsize-1:
                    minibatchsize += extra
                #Where we store the total weight updates for the mini-batch
                self.weight_updates = [np.zeros(layer.w_size) for layer in self.layers]
                #Where we store the total bias updates for the mini-batch
                self.bias_updates = [np.zeros(layer.b_size) for layer in self.layers]
                #Now it's time to update the weights!
                self.update_weights(training,minibatchsize,lrn_rate)

            #We already added the loss for epoch 0 to the history, so only do
            #this for epochs 1 and beyond. Values for trn_output are added
            #in the update_weights() method to save calculation time
            
            if epoch_nr > 0:
                loss_array=ErrorV(self.trn_output,training[1])
                #Calculate the average training loss of the epoch and save it
                self.history['trn'].append(sum(loss_array)/len(loss_array))
            #Get the validation loss for the epoch if that flag is on
            #But always save the validation of the final epoch!
            if save_val or epoch_nr == (epochs - 1):
                # Temporarily turn off dropout (for loss stuff)
                print("\nFinal validation loss (dropout off)") if debugging else None
                for layer in self.layers:
                    layer.disable_dropout()
                    print("dropout_enabled: " + str(layer.dropout_enabled)) if debugging else None

                self.val_output = self.feed_all_patterns(validation[0])
                loss_array=ErrorV(np.array(self.val_output),validation[1])
                #Calculate the average validaiton loss of the epoch and save it
                self.history['val'].append(sum(loss_array)/len(loss_array))

            #Now we start a new epoch!

        print("\nFinal training loss (dropout off)") if debugging else None
        #Calculate the error after the final weight update without updating
        #the weights further to complete the history list
        self.trn_output = self.feed_all_patterns(training[0])
        loss_array=ErrorV(np.array(self.trn_output),training[1])
        #Calculate the average training loss of the epoch and save it
        self.history['trn'].append(sum(loss_array)/len(loss_array))

        #Note that when we calculate the loss for the validation data we
        #actually do it after the weights have been updated for the epoch!
        #Therefore, we DON'T need an extra run through all patterns once
        #we have gone through all epochs like we do for the training data!
        
        if history_plot:
            self.show_history(epochs) #This will generate the "loss plot"


class Layer_Dense:
    # Initialize the dense layer with inputs random weights & biases and
    # the right activation function, and a 
    def __init__(self, dim, nodes, activation, l2_s, dropout_rate = 0, rng = np.random.default_rng()):
        self.weights = rng.standard_normal(size = (nodes, dim))
        self.biases = rng.standard_normal(size = (1, nodes))
        self.w_size = self.weights.shape
        self.b_size = self.biases.shape
        self.activation = activation
        self.l2_s = l2_s
        self.input = None
        self.output = None
        self.enable_dropout()
        self.set_dropout_rate(dropout_rate)
        self.generate_dropout_mask(rng=rng)

    # Enable dropout
    def enable_dropout(self):
        self.dropout_enabled = True

    # Disable dropout
    def disable_dropout(self):
        self.dropout_enabled = False

    # Set dropout rate
    def set_dropout_rate(self, dropout_rate):
        if (dropout_rate == 1):
            warnings.warn("You are using a dropout rate of 1, meaning all nodes are being dropped.")
        self.dropout_rate = dropout_rate

    # Generate the dropout mask using the layers dropout_rate parameters
    # The dropout mask is a numpy array the size of the input layer
    # The dropout mask is not applied to the layer itself, but to the input
    # -- this is due to the initial input layer not being defined as a layer, but needing dropout capability
    def generate_dropout_mask(self, rng = np.random.default_rng()):
        # Get dropout rate
        dropout_rate = self.dropout_rate

        # Get number of inputs, for use in setting the size of the dropout mask
        num_inputs = len(self.weights[0])

        # Generate dropout mask
        is_all_dropped = True
        while is_all_dropped:
            # Generate dropout mask, until the generated mask keeps at least one value
            # (True means drop value, False means keep value)
            self.dropout_mask = rng.choice(a=[False, True], size=num_inputs, p=[1-dropout_rate, dropout_rate])
            is_all_dropped = np.all(self.dropout_mask == True)
            # If dropout_rate is 1, then all MUST be dropped, so don't try to keep at least one value
            if (dropout_rate == 1):
                break

    # Get the input with the dropout applied
    # If dropout is disabled, then a temporary dropout mask which keeps all is used
    def get_input_w_dropout(self):
        dropout_mask = self.dropout_mask
        if not self.dropout_enabled:
            # Do not drop any nodes if dropout is disabled
            dropout_mask = np.full(self.dropout_mask.size, False)
    
        # Create a masked array by applying the dropout_mask to the input
        input_w_dropout = np.ma.MaskedArray(self.input, dropout_mask, fill_value=0) 
        return(input_w_dropout)

    # Calculate the output of the layer
    def calc_output(self, X):
        self.input = X
        # Get the input with the dropout mask applied
        input_w_dropout = self.get_input_w_dropout()
        # Calculate the argument(s) of the layer
        # NOTE that dropout is not applied to biases, but that 
        argument = (np.ma.dot(self.weights, input_w_dropout, strict=True) + self.biases).flatten()
        self.output = self.activation['act'](argument)
        #We want to flatten the output to turn it into a 1D array, otherwise
        #it will be a 2D array which causes problems when we want to check
        #the input dimension to set up the next layer.
        return self.output

def Error(y, d):
    #E = -(np.dot(d,np.log(y))+np.dot((1-d),np.log(1-y)))[0] save for later
    if d == 0:
        return(-np.log(1-y))
    elif d == 1:
        return(-np.log(y))

def classification_loss(y,d):
    return y-d

ErrorV = np.vectorize(Error)

loss = np.vectorize(classification_loss)