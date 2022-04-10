import numpy as np
import matplotlib.pyplot as plt
import warnings

#Remember:
#Weights on the same ROW act on the same node
#Weights on the same COLUMN come from the same node

class Model:
    def __init__(self, input_dim, layer_defines, rng = np.random.default_rng(), is_debugging = False, is_janky = False):
        self.layers = [] #This is where we'll put all the layers we make
        columns = input_dim #Number of columns to use for the first weight matrix
        self.rng = rng
        self.is_debugging = is_debugging
        self.is_janky = is_janky
        print("\ninitialising model...") if self.is_debugging else None
        #Make as many layers as we have defined properties for
        for layer_properties in layer_defines:
            #Create a dense layer
            self.layers.append(Layer_Dense(columns, *layer_properties,
                                           rng = self.rng))
            #Then we want to know how many columns the weight matrix in the
            #next layer should have, by looking at how many rows (i.e. how the
            #many nodes) the current one has.
            columns = self.layers[-1].weights.shape[0]
            print(f"initialised layer {len(self.layers)} with weights {self.layers[-1].weights} and biases {self.layers[-1].biases}.") if self.is_debugging else None
        print("finished initialising model!\n") if self.is_debugging else None

    def feed_forward(self, X):
        print(f"\nbeginning feed forward for pattern {X}...") if self.is_debugging else None
        for i in range(0, len(self.layers)):
            layer = self.layers[i]
            print(f"\nlayer {i}...") if self.is_debugging else None
            layer.input = X
            next_layer_input = layer.calc_output(X)
            print(f"dropout mask = {layer.dropout_mask}") if self.is_debugging else None
            print(f"is dropout enabled = {layer.dropout_enabled}")if self.is_debugging else None
            print(f"input w/ dropout = {layer.get_input_w_dropout()}") if self.is_debugging else None
            print(f"bias = {layer.biases}") if self.is_debugging else None
            print(f"output = {str(next_layer_input)}") if self.is_debugging else None
            X = next_layer_input
        print("\nfinished feed forward.") if self.is_debugging else None

    def feed_all_patterns(self, patterns):
        outputs = []
        for pattern in patterns:
            self.feed_forward(pattern) #Update the network's output...
            outputs.append(float(self.layers[-1].output)) #...and save it
        return np.array(outputs)

    def enable_all_dropout(self):
        for layer in self.layers:
            layer.enable_dropout()

    def disable_all_dropout(self):
        for layer in self.layers:
            layer.disable_dropout()

    def backpropagate(self, y, d):
        print("\nbeginning backpropagation...") if self.is_debugging else None
        print(f"output = {y}") if self.is_debugging else None
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
        deltas = loss(y, d)
        #Now we want to calculate the update for all weights, layer by layer
        for i in range(0, len(self.layers)):
            layer = self.layers[i]
            print(f"layer {len(self.layers) - i}...") if self.is_debugging else None

            #For the current layer we need to know the weights
            #Previous output refers to the layer that comes before in the
            #feed-forward step
            current = layer.weights
            prev_output = layer.get_input_w_dropout()
            print(f"dropout mask = {layer.dropout_mask}") if self.is_debugging else None
            print(f"is dropout disabled = {layer.dropout_enabled}")if self.is_debugging else None
            print(f"input w/ dropout = {layer.get_input_w_dropout()}") if self.is_debugging else None
            print(f"bias = {layer.biases}") if self.is_debugging else None

            #Make an update matrix for the current layer
            w_update = np.ma.zeros(current.shape)
            b_update = np.ma.zeros(current.shape[0])

            #Equation 2.11 in FYTN14 Lecture Notes
            for row in range(0,current.shape[0]):
                w_update[row] = np.ma.dot(deltas[row], prev_output, strict=True)
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
        print("finished backpropagation!") if self.is_debugging else None
        return all_w_updates, all_b_updates

    def update_weights(self, trn, minibatchsize, lrn_rate):
        x_trn, d_trn = trn
        print("\nbegin updating weights and biases for this epoch...") if self.is_debugging else None
        print("\nbegin calculating weight and bias updates for this epoch...") if self.is_debugging else None
        #Now go through each pattern in the mini-batch
        for n in range(0, minibatchsize):
            print(f"\ncalculating weight updates for pattern {n}...") if self.is_debugging else None
            if not self.is_janky:
                # Regenerate the dropout masks for each new pattern
                for i in range(0, len(self.layers)):
                    layer = self.layers[i]
                    print(f"\ngenerating dropout mask for layer {i}...") if self.is_debugging else None
                    layer.generate_dropout_mask(rng=self.rng)
                    print(f"generated dropout mask {layer.dropout_mask}.") if self.is_debugging else None

            # This loop is dedicated to determining how many times each weight is used (not dropped)
            for i in range(0, len(self.layers)):
                layer = self.layers[i]

                dropout_mask = layer.dropout_mask # The dropout mask which is applied to the inputs of the current layer
                next_dropout_mask = None
                if i != len(self.layers)-1:
                    # If it is not the last layer
                    next_dropout_mask = self.layers[i+1].dropout_mask[np.newaxis] # The dropout mask which is applied to the nodes of the current layer
                else:
                    # If it is the last layer
                    next_dropout_mask = np.full(layer.b_size, False) # The final dropout mask should just be a false value for each node, which shares the same shape as the bias size

                next_dropout_mask = next_dropout_mask

                dropout_mask_weights = np.tile(dropout_mask, (layer.w_size[0], 1))
                next_dropout_mask_weights = np.tile(next_dropout_mask.T, (1, layer.w_size[1]))

                next_dropout_mask_biases = np.reshape(next_dropout_mask, layer.b_size)

                num_weight_updates = np.zeros(layer.w_size, dtype=int)
                num_bias_updates = np.zeros(layer.b_size, dtype=int)

                # Combine the dropout with the next dropout mask for weights
                for j in range(0, len(dropout_mask_weights)):
                    for k in range(0, len(dropout_mask_weights[j])):
                        if dropout_mask_weights[j, k] == False and next_dropout_mask_weights[j, k] == False:
                            num_weight_updates[j, k] = 1

                self.num_weight_updates[i] = np.add(self.num_weight_updates[i], num_weight_updates)

                # Just use the next dropout mask for biases
                for j in range(0, len(next_dropout_mask_biases)):
                    for k in range(0, len(next_dropout_mask_biases[j])):
                        if next_dropout_mask_biases[j, k] == False:
                            num_bias_updates[j, k] = 1

                self.num_bias_updates[i] = np.add(self.num_bias_updates[i], num_bias_updates)           

            #Find the network's output for the pattern
            self.feed_forward(x_trn[n])

            #Adding updates for every pattern to the total weight updates
            all_w_updates, all_b_updates = self.backpropagate(self.layers[-1].output, d_trn[n])

            print(f"\nweight updates for this pattern (in reverse order) = {all_w_updates}") if self.is_debugging else None
            print(f"bias updates for this pattern (in reverse order) = {all_b_updates}") if self.is_debugging else None
            print(f"\nfinished calculating weight updates for this pattern!") if self.is_debugging else None

            #We want it in the reverse order because the all_updates we get from
            #the backpropagation will be in the reverse order
            self.weight_updates.reverse() ; self.bias_updates.reverse()
            for i in range(0, len(self.layers)):
                self.weight_updates[i] += all_w_updates[i]
                self.bias_updates[i] += all_b_updates[i]
            # Reverse back to normal order
            self.weight_updates.reverse() ; self.bias_updates.reverse()

        #Now we have all of the weight updates for the current mini-batch!
        print(f"\nweight updates for this epoch = {self.weight_updates}") if self.is_debugging else None
        print(f"bias updates for this epoch = {self.bias_updates}") if self.is_debugging else None
        print(f"\nfinished calculating weight updates for this epoch!") if self.is_debugging else None

        print(f"\nbegin applying weight updates...") if self.is_debugging else None
        #Actually update the weights
        for i in range(0, len(self.layers)):
            layer = self.layers[i] #Current layer
            #Update the weights with the result from backpropagation and
            #the derivative of the L2-term
            print(f"\napplying weight updates for layer {i}...") if self.is_debugging else None
            print("\nbefore:") if self.is_debugging else None
            print(f"weights = {layer.weights}") if self.is_debugging else None
            print(f"biases = {layer.biases}") if self.is_debugging else None

            if self.is_janky:
                self.num_weight_updates[i] = np.full_like(self.num_weight_updates[i], minibatchsize)
                self.num_bias_updates[i] = np.full_like(self.num_bias_updates[i], minibatchsize)

            mean_weight_updates = np.divide(self.weight_updates[i], self.num_weight_updates[i], out=np.zeros_like(self.weight_updates[i]), where=self.num_weight_updates[i]!=0)
            actual_mean_weight_updates = -(lrn_rate*mean_weight_updates + layer.l2_s*layer.weights)
            layer.weights += actual_mean_weight_updates
            mean_bias_updates = np.divide(self.bias_updates[i], self.num_bias_updates[i], out=np.zeros_like(self.bias_updates[i]), where=self.num_bias_updates[i]!=0)
            actual_mean_bias_updates = -(lrn_rate*mean_bias_updates)
            layer.biases += actual_mean_bias_updates

            print("\nthe update:") if self.is_debugging else None
            print(f"num weight updates = {self.num_weight_updates[i]}") if self.is_debugging else None
            print(f"num bias updates = {self.num_bias_updates[i]}") if self.is_debugging else None
            print(f"weight updates (reduced by learning rate and L2) = {actual_mean_weight_updates}") if self.is_debugging else None
            print(f"bias updates (reduced by learning rate and L2) = {actual_mean_bias_updates}") if self.is_debugging else None

            print("\nafter:") if self.is_debugging else None
            print(f"weights = {layer.weights}") if self.is_debugging else None
            print(f"biases = {layer.biases}") if self.is_debugging else None

        print("\nfinished updating weights!") if self.is_debugging else None

    def show_history(self, epochs):
        if epochs > len(self.history['trn']):
            print("WARNING: Attempted to plot with more epochs than saved, check that you trained with 'should_save_intermediary_history = True'. Only use 'should_save_intermediary_history = False' when you want to be extra efficient!")
            return None

        # Plot the error over all epochs
        plt.figure()
        plt.plot(np.arange(0, epochs+1), self.history['trn'], 'orange', label='Training error')
        plt.plot(np.arange(0, epochs+1), self.history['val'], 'blue', label='Validation error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Error over epochs')
        plt.legend()
        plt.draw()

    def save_history(self, trn, val):
        # Save the training and validation loss for the current epoch
        x_trn, d_trn = trn
        x_val, d_val = val

        N = len(x_trn) #number of patterns in training data
        N_val = len(x_val) #number of patterns in validation data

        # Temporarily disable dropout (for loss stuff)
        print("dropout disabled (whilst calculating loss).") if self.is_debugging else None
        self.disable_all_dropout()  

        print("\ncalculating training loss...") if self.is_debugging else None
        #Get the training loss
        trn_outputs = self.feed_all_patterns(x_trn)
        loss_array = ErrorV(np.array(trn_outputs), d_trn)
        mean_loss = sum(loss_array)/N
        self.history['trn'].append(mean_loss)
        print(f"\ntraining loss = {self.history['trn'][-1]}") if self.is_debugging else None

        print("\ncalculating validation loss...") if self.is_debugging else None
        #Get the validation loss
        val_outputs = self.feed_all_patterns(x_val)
        loss_array = ErrorV(np.array(val_outputs), d_val)
        mean_loss = sum(loss_array)/N_val
        self.history['val'].append(mean_loss)
        print(f"\nvalidation loss = {self.history['val'][-1]}") if self.is_debugging else None

        # Re-enable dropout for learning
        print("\ndropout re-enabled.") if self.is_debugging else None
        self.enable_all_dropout()

    def train(self, trn, val, lrn_rate, epochs, minibatchsize=0,
              should_save_intermediary_history=True):
        print("begin training...") if self.is_debugging else None
        x_trn, d_trn = trn
        x_val, d_val = val

        N = len(x_trn) #number of patterns in training data
        N_val = len(x_val) #number of patterns in validation data

        #If minibatchsize is 0 (i.e. default), do regular gradient descent
        if minibatchsize == 0:
            minibatchsize = N

        fixed_minibatchsize = minibatchsize #Save the true mini-batch size somewhere
        #Check if the mini-batch size is larger than the total amount of training patterns

        if minibatchsize > N:
            raise Exception(f'Your mini-batch size is too large! With the current training data it cannot exceed {N}.')

        #Number of extra patterns to be added onto the final mini-batch
        extra = N % fixed_minibatchsize
        
        self.history = {'trn':[],'val':[]} #This is where we will save the loss after each epoch

        print("\ncalculating initial training and validation loss...") if self.is_debugging else None
        self.save_history(trn, val) # Always save the initial history
        print("\nfinished calculating initial training and validation loss!") if self.is_debugging else None

        #This loop is for going through the desired amount of epochs
        for epoch_nr in range(0, epochs):
            print(f"\nbegin epoch {epoch_nr}...") if self.is_debugging else None

            print(f"pattern order shuffled.") if self.is_debugging else None
            p = self.rng.permutation(len(x_trn))
            training = [x_trn[p], d_trn[p]]
            #Restore the mini-batch size to the original value
            minibatchsize = fixed_minibatchsize

            # Regenerate the dropout masks for each epoch
            if self.is_janky:
                for i in range(0, len(self.layers)):
                    layer = self.layers[i]
                    print(f"\ngenerating dropout mask for layer {i}...") if self.is_debugging else None
                    layer.generate_dropout_mask(rng=self.rng)
                    print(f"generated dropout mask {layer.dropout_mask}.") if self.is_debugging else None

            #Now it's time to go through all of the mini-batches
            for minibatch_nr in range(0,N//fixed_minibatchsize):
                #This makes sure the last minibatch gets any remaining patterns
                if minibatch_nr == N//fixed_minibatchsize-1:
                    minibatchsize += extra

                #Where we store the total weight and bias updates for the mini-batch
                self.weight_updates = [np.zeros(layer.w_size) for layer in self.layers]
                self.bias_updates = [np.zeros(layer.b_size) for layer in self.layers]
                # Where we store the total number of updates for the mini-batch
                self.num_weight_updates = [np.zeros(layer.w_size, dtype = int) for layer in self.layers]
                self.num_bias_updates = [np.zeros(layer.b_size, dtype = int) for layer in self.layers]
                print(f"\initialised empty weight updates {self.weight_updates} and bias updates {self.bias_updates}.") if self.is_debugging else None
                print(f"initialised empty weight update counters {self.num_weight_updates} and bias update counters {self.num_bias_updates}") if self.is_debugging else None
                
                #Now it's time to update the weights!
                self.update_weights(training, minibatchsize, lrn_rate)

            if should_save_intermediary_history and epoch_nr != epochs-1:
                print("\ncalculating intermediary training and validation loss...") if self.is_debugging else None
                self.save_history(trn, val)
                print("finished calculating intermediary training and validation loss!") if self.is_debugging else None

        print("\ncalculating final training and validation loss...") if self.is_debugging else None
        self.save_history(trn, val) # Always save the final history
        print("finished calculating final training and validation loss!") if self.is_debugging else None

        print("\nfinished training!") if self.is_debugging else None


class Layer_Dense:
    # Initialize the dense layer with random weights & biases, and
    # the right activation function and dropout rate
    def __init__(self, dim, nodes, activation, l2_s, dropout_rate = 1, rng = np.random.default_rng()):
        self.weights = rng.normal(0, 1/np.sqrt(dim), size = (nodes, dim))
        #self.weights = rng.standard_normal(size = (nodes, dim)) # old
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
        if (dropout_rate == 0):
            warnings.warn("You are using a dropout rate of 0, meaning all nodes are being dropped.")
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
            self.dropout_mask = rng.choice(a=[False, True], size=num_inputs, p=[dropout_rate, 1-dropout_rate])
            is_all_dropped = np.all(self.dropout_mask == True)
            # If dropout_rate is 0, then all MUST be dropped, so don't try to keep at least one value
            if (dropout_rate == 0):
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
        return(-np.log(1 - y))
    elif d == 1:
        return(-np.log(y))

def classification_loss(y, d):
    return y - d

ErrorV = np.vectorize(Error)

loss = np.vectorize(classification_loss)