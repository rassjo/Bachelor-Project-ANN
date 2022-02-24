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
    
#For lambda
start_la = 0
final_la = 0.01
number_to_try = 50

#For number of hidden nodes, stepsize is currently 1
startNumberHidden = 10
endNumberHidden = 30

#For the model
learnRate = 0.1
epochs = 1000
minibatchsize = 0 #0 if we don't want to use minibatches

extra_patterns = 0
lambda_x = np.linspace(start_la,final_la,number_to_try)
nHidden = []
bestLambd = []

for hid in range(startNumberHidden,endNumberHidden+1): #range for the numbers of hidden nodes we want to try
    trainy = []
    valy = []
    lambd_to_cms = {}
    for lambd in lambda_x: #number of lambdas we want to test for each number of hidden nodes
        #------------------------------------------------------------------------------
        # Create random number generators:
        # seed == -1 for random rng, seed >= 0 for fixed rng (seed should be integer)
        data_seed = 1
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
    
        x_trn = trn[0]
        d_trn = trn[1]
    
        x_val = val[0]
        d_val = val[1]
    
        #Properties of all the layers
        #Recipe for defining a layer: [number of nodes, activation function, L2]
        layer_defines = [[hid, act.tanh, lambd],
                         [1, act.sig, lambd]]
    
        test = ann.Model(input_dim, layer_defines, ann_rng)
    
        #Check results
        answer1 = check_results(test, False)
    
        outputs = test.feed_all_patterns(x_trn) # Collect the outputs for all the inputs
        outputs = test.feed_all_patterns(x_val) # Collect the outputs for all the inputs
    
        plt.show()
    
        test.train(trn,val,learnRate,epochs,minibatchsize) #training, validation, lrn_rate, epochs, minibatchsize=0
    
        #Check results again
        answer2 = check_results(test, False)
    
        outputs = test.feed_all_patterns(x_trn) #Collect the outputs for all the inputs
        cm_trn = cs.construct_confusion_matrix(outputs, d_trn)
    
        outputs = test.feed_all_patterns(x_val) #Collect the outputs for all the inputs
        cm_val = cs.construct_confusion_matrix(outputs, d_val)
    
        plt.show()
    
        validation = test.history['val'][-1] #Loss for validation
    
        # Display losses
        print("Lambda", lambd)
        print("Loss before training", answer1)
        print("Loss after training", answer2)
        print("Validation loss", test.history['val'][-1])
        print("")
        
        trainy.append(answer2)
        valy.append(validation)
        
        lambd_to_cms[lambd] = {"trn": cm_trn, "val": cm_val}
    
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
    plt.savefig(f'LambdaPlot_{hid}_hidden_nodes.png')
    plt.show()
    
    # Construct a list of training and validation accuracies from the lambd_to_cms dictionary
    acc_trn = []
    acc_val = []
    for lambd in lambda_x:
        stats_trn = cs.calculate_stats(lambd_to_cms[lambd]["trn"])
        acc_trn.append(stats_trn[0]["advanced"]["acc"])
        stats_val = cs.calculate_stats(lambd_to_cms[lambd]["val"])
        acc_val.append(stats_val[0]["advanced"]["acc"])
        
    # Plot the accuracy over lambda plot
    plt.figure()
    plt.plot(lambda_x, acc_trn, "go", label="Training")
    plt.plot(lambda_x, acc_val, "go", label="Validation")
    plt.xlabel("$\lambda$")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over " + "$\lambda$")
    plt.legend()
    plt.savefig(f'lambda_accuracy_plot_{hid}.png')
    plt.show()

#The best lambda for each number of hidden nodes is plotted
plt.figure()
plt.plot(nHidden, bestLambd, 'ro', label='best lambdas vs # hidden nodes (1 layer)')
plt.xlabel('# hidden nodes')
plt.ylabel('best lambdas')
plt.title('best lambdas vs # hidden nodes (1 layer)')
plt.legend()
plt.savefig('HyperparameterPlot.png')
plt.show()



