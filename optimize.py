import ANN as ann
import numpy as np
import activation_functions as act
import synthetic_data_generation as sdg
import classification_statistics as cs
import matplotlib.pyplot as plt

def check_results(model, show=False):
    loss = 0
    N = len(trn[0])
    for n in range(0,N):
        model.feed_forward(trn[0][n])
        #print("Pattern", n ," in: ", trn[0][n])
        if show:
            print("Pattern", n ,"out:", model.layers[-1].output)
            print("Pattern", n ,"targ:", trn[1][n])
        loss += ann.Error(model.layers[-1].output, trn[1][n])
    return loss/N

def check_layers(model):
    weights = [layer.weights for layer in model.layers]
    biases = [layer.biases for layer in model.layers]
    print("Weights:", weights)
    print("Biases:", biases)

lambdx = []
#notrainy = []
trainy = []
valy = []
for la in range(0, 10):
    #------------------------------------------------------------------------------
    # Create random number generators:
    # seed == -1 for random rng, seed >= 0 for fixed rng (seed should be integer)
    data_seed = 5
    ann_seed = data_seed

    def generate_rng(seed):
        # seed should be an integer
        # -1 is for random rng, integer >= 0 for fixed
        if (seed != -1):
            return np.random.default_rng(data_seed)
        return np.random.default_rng()

    data_rng = generate_rng(data_seed)
    ann_rng = generate_rng(ann_seed)

    #------------------------------------------------------------------------------
    # Import data
    trn, val = sdg.generate_datasets('circle_ception', try_plot = True,
                                     rng = data_rng)
    input_dim = len(trn[0][0]) #Get the input dimension from the training data

    # Just use the training for the moment
    x_trn = trn[0]
    d_trn = trn[1]

    x_val = val[0]
    d_val = val[1]

    lambd = 0 + la*0.0025 #(2 inputs, 200 patterns -> conts*1/100 as size for lambda optimally? (for as good validation performance as possible)) 

    print(lambd)
    #Properties of all the layers
    #Recipe for defining a layer: [number of nodes, activation function, L2]
    layer_defines = [[20, act.tanh, lambd],
                     [20, act.tanh, lambd],
                     [20, act.tanh, lambd],
                     [1, act.sig, lambd]]

    test = ann.Model(input_dim, layer_defines, ann_rng)

    #Check results
    answer1 = check_results(test, False)

    outputs = test.feed_all_patterns(x_trn) # Collect the outputs for all the inputs
    statistics = cs.stats(outputs, d_trn, f'pre-training {lambd}', should_plot_cm = True)

    outputs = test.feed_all_patterns(x_val) # Collect the outputs for all the inputs
    cs.stats(outputs, d_val, f'pre-validation {lambd}', should_plot_cm = True)

    plt.show()

    test.train(trn,val,0.1,40,0) #training, validation, lrn_rate, epochs, minibatchsize=0

    #Check results again
    answer2 = check_results(test, False)

    outputs = test.feed_all_patterns(x_trn) #Collect the outputs for all the inputs
    cs.stats(outputs, d_trn, f'post-training {lambd}', should_plot_cm = True)

    outputs = test.feed_all_patterns(x_val) #Collect the outputs for all the inputs
    cs.stats(outputs, d_val, f'post-valdation {lambd}', should_plot_cm = True)

    plt.show()

    validation = test.history['val'][-1] #Loss for validation

    # Display losses
    print("\nLoss before training", answer1)
    print("Loss after training", answer2)
    print("Validation loss", test.history['val'][-1])

    lambdx.append(lambd)
    #notrainy.append(answer1)
    trainy.append(answer2)
    valy.append(validation)

plt.figure()
#plt.plot(lambdx, notrainy, 'ro', label='error before train vs lambd') #consider using (x,y,linestyle = 'none',marker = '.',markersize= 0.1)
plt.plot(lambdx, trainy, 'go', label='error after train vs lambd')
plt.plot(lambdx, valy, 'bo', label='Validation error vs lambd')
plt.xlabel('lambdas')
plt.ylabel('Error')
plt.title('Error over lambdas')
plt.legend()
plt.savefig('ErrorPlot.png')
plt.show()
