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

start_la = 0
final_la = 0.1
number_to_try = 3

extra_patterns = 0

lambda_x = np.linspace(start_la,final_la,number_to_try)
nHidden = []
bestLambd = []

for hid in range(1,10): #range for the numbers of hidden nodes we want to try
    trainy = []
    valy = []
    for lambd in lambda_x: #number of lambdas we want to test for each number of hidden nodes
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
        trn, val = sdg.generate_datasets('lagom',
                                     extra = extra_patterns,
                                     val_mul = 10,
                                     try_plot = True,
                                     rng = data_rng)
        input_dim = len(trn[0][0]) #Get the input dimension from the training data
    
        # Just use the training for the moment
        x_trn = trn[0]
        d_trn = trn[1]
    
        x_val = val[0]
        d_val = val[1]
    
        print(lambd)
        #Properties of all the layers
        #Recipe for defining a layer: [number of nodes, activation function, L2]
        layer_defines = [[hid, act.tanh, lambd],
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
        
        trainy.append(answer2)
        valy.append(validation)
        
        extra_patterns += 0
    
    #Appending the number of hidden nodes and best lambda for that number of nodes, to be plotted later
    nHidden.append(hid)

    #I'm appending the lambda that gives the lowest validation-, and not training-, loss, as that would be zero
    bestLambd.append(lambda_x[valy.index(min(valy))]) 
    
    #The loss for each lambda is plotted once for every new number of hidden nodes
    plt.figure()
    plt.plot(lambda_x, trainy, 'go', label='error after train vs lambd')
    plt.plot(lambda_x, valy, 'bo', label='Validation error vs lambd')
    plt.xlabel('lambdas')
    plt.ylabel('Error')
    plt.title('Error over lambdas')
    plt.legend()
    #plt.savefig('ErrorPlot.png')
    plt.show()

#The best lambda for each number of hidden nodes is plotted
plt.figure()
plt.plot(nHidden, bestLambd, 'ro', label='best lambdas vs # hidden nodes (1 layer)')
plt.xlabel('# hidden nodes')
plt.ylabel('best lambdas')
plt.title('best lambdas vs # hidden nodes (1 layer)')
plt.legend()
#plt.savefig('LambdaPlot.png')
plt.show()



