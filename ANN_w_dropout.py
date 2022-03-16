import numpy as np
import activation_functions as act
import synthetic_data_generation as sdg
import classification_statistics as cs
import matplotlib.pyplot as plt

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
            self.layers.append(Layer_Dense(columns,*layer_properties,
                                           rng = rng))
            #Then we want to know how many columns the weight matrix in the
            #next layer should have, by looking at how many rows (i.e. how the
            #many nodes) the current one has.
            columns = self.layers[-1].weights.shape[0]

    def feed_forward(self, X):
        for layer in self.layers:
            print("          NEW LAYER:")
            layer.input = X
            next_layer_input = layer.calc_output(X)
            X = next_layer_input

    def feed_all_patterns(self, patterns):
        print("")
        print("FEED FORWARD:")

        outputs = []
        for pattern in patterns:
            print("     NEW PATTERN:")

            self.feed_forward(pattern) #Update the network's output...
            outputs.append(float(self.layers[-1].output)) #...and save it
        return np.array(outputs)

    def backpropagate(self, y, d):
        print("")
        print("BACK PROPAGATE:")
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
            #prev_output = self.layers[i].input
            prev_output = self.layers[i].get_input_w_dropout()

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
            #print("b_update: " + str(b_update))
            #print("w_update: " + str(w_update))

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
                print("     NEW LAYER:")
                self.weight_updates[i] += all_w_updates[i]
                self.bias_updates[i] += all_b_updates[i]
                print("dropout mask: " + str(self.layers[i].dropout_mask))
                print("weight update: " + str(self.weight_updates[i]))
                print("bias update:" + str(self.bias_updates[i]))
            #print("weight updates: " + str(self.weight_updates))
            #print("bias updates:" + str(self.bias_updates))
            #Save the output of each pattern for calculating the loss later
            self.trn_output[self.n] = float(self.layers[-1].output)
            self.n += 1 #Increment the counter when we go to the next pattern

        #Now we have all of the weight updates for the current mini-batch!

        #Reverse the layers because weight_updates is reversed
        self.layers.reverse()
        #Actually update the weights
        for i in range(0, len(self.layers)):
          layer = self.layers[i] #Current layer
          #Update the weights with the result from backpropagation and
          #the derivative of the L2-term
          layer.weights -= (lrn_rate*self.weight_updates[i]/minibatchsize + layer.l2_s*layer.weights)
          layer.biases -= lrn_rate*self.bias_updates[i]/minibatchsize
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

        #Get the initial training loss
        self.trn_output = self.feed_all_patterns(training[0])
        loss_array=ErrorV(np.array(self.trn_output),training[1])
        self.history['trn'].append(sum(loss_array)/N)

        #Get the initial validation loss
        self.val_output = self.feed_all_patterns(validation[0])
        loss_array=ErrorV(np.array(self.val_output),validation[1])
        self.history['val'].append(sum(loss_array)/N_val)

        #This loop is for going through the desired amount of epochs
        for epoch_nr in range(0, epochs):
            #This is where we save results after each training pattern
            self.trn_output = np.zeros(N)
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
                self.val_output = self.feed_all_patterns(validation[0])
                loss_array=ErrorV(np.array(self.val_output),validation[1])
                #Calculate the average validaiton loss of the epoch and save it
                self.history['val'].append(sum(loss_array)/len(loss_array))

            #Now we start a new epoch!

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
    #Initialize the dense layer with inputs random weights & biases and
    #the right activation function
    def __init__(self, dim, nodes, activation, l2_s, dropout_rate=1, rng=np.random.default_rng()):
        self.weights = rng.standard_normal(size = (nodes, dim))
        self.biases = rng.standard_normal(size = (1, nodes))
        self.w_size = self.weights.shape
        self.b_size = self.biases.shape
        self.activation = activation
        self.l2_s = l2_s
        self.input = None
        self.output = None
        self.dropout_rate = dropout_rate # probability that a given input node is dropped
        self.refresh_dropout_mask(rng)

    def refresh_dropout_mask(self, rng=np.random.default_rng()):
        # Mask is the same length as the number of inputs, it is applied to each input. (essentially acts on the previous layer)
        num_inputs = len(self.weights[0])
        self.dropout_mask = rng.choice(a=[True, False], size=num_inputs, p=[1-self.dropout_rate, self.dropout_rate]) # True means drop value, False means keep value.

    def get_input_w_dropout(self):
        input_w_dropout = np.ma.MaskedArray(self.input, self.dropout_mask, fill_value=0) 
        return(input_w_dropout)

    #Calculate the output of the layer
    def calc_output(self, X):
        self.input = X

        input_w_dropout = self.get_input_w_dropout()
        argument_w_dropout = (np.ma.dot(self.weights, input_w_dropout, strict=True) + self.biases).flatten()
        #argument = (np.dot(self.weights, self.input) + self.biases).flatten()

        #print("a:"+ str(argument))
        print("dropout mask: " + str(self.dropout_mask))
        print("argument w dropout: " + str(argument_w_dropout))

        self.output = self.activation['act'](argument_w_dropout)
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