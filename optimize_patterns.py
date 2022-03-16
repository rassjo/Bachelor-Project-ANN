import ANN as ann
import numpy as np
import activation_functions as act
import synthetic_data_generation as sdg
import classification_statistics as cs
import matplotlib.pyplot as plt
from id_generator import best_hash
import sys
import warnings

def write_hyperparameters(name,hp,identifier,seed):
    # If the document already exists, then do nothing.
    try:
        open(name, "x")
    except:
        raise Exception(f"Results from a run with these settings already exists! Please delete {name}!")
    
    # Overwrite the empty file with the hyperparameters
    with open(name, "w") as f:
        print(f"Hyperparameters: {hp}\n", file=f)
        print(f"Seed: {seed}\n", file=f)
        print(f"ID: {identifier}\n", file=f)
        print("Below are the results!\n", file=f)
        print("Best Lambda ; Number of training patterns\n", file=f)

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
number_to_try = 5

#For patterns
numPatterns = 4 #how many (extra) times we make a new number of patterns

#For the model
minibatchsize = 0 #0 if we don't want to use minibatches

lambda_x = [0.00001*10**(int(0.25*i))*2**(i%4) for i in range(0,number_to_try)]

hp = {'lrn_rate': 0.1,
      'epochs': 10,
      'lambdas': lambda_x,
      'val_mul': 10,
      'hidden': 15,
      'dataset': 'lagom'}

# For inputting seed from the command line. (i.e. use ```nohup nice n -19 python3 optimize_patterns.py seed &```, where seed is an integer).
seed = -1
try:
  seed = int(sys.argv[1])
except TypeError:
  warnings.warn("Seed input was not an integer. Using random seed instead.")
except:
  warnings.warn("Seed input was missing. Using random seed instead.")

#For hyperparameters in .txt

hyperparameter_string = str(list(hp.values()))
identifier = best_hash(hyperparameter_string)
uniqueFileName = "patterns_lambda_" + str(identifier) + "_" + str(seed) + ".txt"

write_hyperparameters(uniqueFileName,hp,identifier,seed)

nPatterns = []
bestLambd = []

if minibatchsize:
    raise Exception('Turn off stochastic gradient descent!')

for i in range(0,numPatterns+1): #range for the numbers of patterns
    trainy = []
    valy = []
    lambd_to_cms = {}
    extra_patterns = 4*10**(int(0.25*i))*2**(i%4) #patterns = constx10^(int(0.25xtrial))x2^(trial%4)
    
    for lambd in lambda_x: #number of lambdas we want to test for each number of hidden nodes
        #------------------------------------------------------------------------------
        # Create random number generators:
        # seed == -1 for random rng, seed >= 0 for fixed rng (seed should be integer)
        data_seed = seed
        ann_seed = data_seed
    
        def generate_rng(seed):
            # seed should be an integer
            # -1 is for random rng, integer >= 0 for fixed
            if (seed != -1):
                return np.random.default_rng(data_seed)
            else:
                raise Exception(f'Please set a fixed seed, otherwise results will not be reproducable!')
                
        data_rng = generate_rng(data_seed)
        ann_rng = generate_rng(ann_seed)
    
        #------------------------------------------------------------------------------
        # Import data
        trn, val = sdg.generate_datasets(hp["dataset"],
                                     extra = extra_patterns,
                                     val_mul = hp["val_mul"],
                                     try_plot = False,
                                     rng = data_rng)
        input_dim = len(trn[0][0]) #Get the input dimension from the training data
    
        x_trn = trn[0]
        d_trn = trn[1]
    
        x_val = val[0]
        d_val = val[1]
    
        #Properties of all the layers
        # Recipe for defining a layer: [number of nodes, activation function, L2, dropout]
        dont_drop_rate = 0.8 # State the probability of KEEPING (because of convention) each node in a layer
        layer_defines = [[hp["hidden"], act.tanh, lambd, dont_drop_rate],
                        [1, act.sig, lambd, dont_drop_rate]]
        test = ann.Model(input_dim, layer_defines, ann_rng)
    
        #Check results
        answer1 = check_results(test, False)
    
        outputs = test.feed_all_patterns(x_trn) # Collect the outputs for all the inputs
        outputs = test.feed_all_patterns(x_val) # Collect the outputs for all the inputs
    
        # plt.show()
        # plt.clf()
    
        test.train(trn,val,hp["lrn_rate"],hp["epochs"],minibatchsize) #training, validation, lrn_rate, epochs, minibatchsize=0
    
        #Check results again
        answer2 = check_results(test, False)
    
        outputs = test.feed_all_patterns(x_trn) #Collect the outputs for all the inputs
        cm_trn = cs.construct_confusion_matrix(outputs, d_trn)
    
        outputs = test.feed_all_patterns(x_val) #Collect the outputs for all the inputs
        cm_val = cs.construct_confusion_matrix(outputs, d_val)
    
        # plt.show()
        # plt.clf()
    
        validation = test.history['val'][-1] #Loss for validation
    
        # Display losses
        # print("Lambda", lambd)
        # print("Loss before training", answer1)
        # print("Loss after training", answer2)
        # print("Validation loss", test.history['val'][-1])
        # print("")
        
        trainy.append(answer2)
        valy.append(validation)
        
        lambd_to_cms[lambd] = {"trn": cm_trn, "val": cm_val}
        

    #Appending the number of patterns nodes and best lambda for that number of nodes, to be plotted later
    nPatterns.append(len(trn[0]))
    
    #I'm appending the lambda that gives the lowest validation-, and not training-, loss, as that would be zero
    bestLambd.append(lambda_x[valy.index(min(valy))]) 
    
    # Write the "best lambda ; number of patterns" pairs
    # In the case of the test crashing, we should adapt the testing loop to start off from where the
    # previous highest number of patterns was
    with open(uniqueFileName, "a") as f:
        f.write("$ " + str(lambda_x[valy.index(min(valy))]) + " ; " + str(len(trn[0])) + "\n")
    
    #The loss for each lambda is plotted once for every new number of patterns
    # plt.figure()
    # plt.plot(lambda_x, trainy, 'go', label='error after train vs lambd')
    # plt.plot(lambda_x, valy, 'bo', label='Validation error vs lambd')
    # plt.xlabel('lambdas')
    # plt.ylabel('Error')
    # plt.title(f'Error over lambdas for {extra_patterns} extra patterns')
    # plt.legend()
    # plt.savefig(f'error_lambda_plot_{extra_patterns}_extra_patterns.png')
    # plt.show()
    # plt.clf()
    
    # Construct a list of training and validation accuracies from the lambd_to_cms dictionary
    acc_trn = []
    acc_val = []
    for lambd in lambda_x:
        stats_trn = cs.calculate_stats(lambd_to_cms[lambd]["trn"])
        acc_trn.append(stats_trn[0]["advanced"]["acc"])
        stats_val = cs.calculate_stats(lambd_to_cms[lambd]["val"])
        acc_val.append(stats_val[0]["advanced"]["acc"])
        
    # Plot the accuracy over lambda plot
    # plt.figure()
    # plt.plot(lambda_x, acc_trn, "go", label="Training")
    # plt.plot(lambda_x, acc_val, "bo", label="Validation")
    # plt.xlabel("$\lambda$")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy over " + "$\lambda$" + " for " + str(extra_patterns) + " extra patterns")
    # plt.legend()
    # plt.savefig(f'accuracy_lambda_plot_{extra_patterns}_extra_patterns.png')
    # plt.show()
    # plt.clf()

#The best lambda for each number of patterns is plotted
# plt.figure()
# plt.errorbar(nPatterns, bestLambd, yerr=[0.1,0.2,0.3,0.4,0.5], fmt = 'o', color='r', ecolor='b', capsize=5, label='best lambdas vs # patterns')
# plt.xlabel('# patterns')
# plt.ylabel('best lambdas')
# plt.title('best lambdas vs # patterns')
# plt.legend()
# plt.savefig('patterns_lambda_plot_' + str(seed) +'.png')
# plt.show()
# plt.clf()
