import ANN as ann
import numpy as np
import activation_functions as act
import synthetic_data_generation as sdg
import classification_statistics as cs
import matplotlib.pyplot as plt
import sys
import warnings
import random as rand
from id_generator import best_hash

def write_hyperparameters(name,hp,identifier,seed):
    # In case we are running with the same settings, we append the values for
    # the hyperparameter we have already tried so we don't redo them later
    tried_values = []
    try:
        open(name, "x")
    except:
        with open(name, 'r') as data:
            for line in data:
                if line[0] != '$':
                    continue
                values = line.split(';')
                tried_values.append(float(values[1][1:-1]))
        return tried_values
    
    # Overwrite the empty file with the hyperparameters
    with open(name, "w") as f:
        print(f"Hyperparameters: {hp}\n", file=f)
        print(f"Seed: {seed}\n", file=f)
        print(f"ID: {identifier}\n", file=f)
        print("Below are the results!\n", file=f)
        print("Best Lambda ; Number of training patterns\n", file=f)
    return tried_values

#For lambda
number_to_try = 3
lambda_x = [0.00001*10**(int(0.25*i))*2**(i%4) for i in range(0,number_to_try)]

#For patterns
numPatterns = 5 #how many (extra) times we make a new number of patterns
extra_patterns = [4*10**(int(0.25*i))*2**(i%4) for i in range(0,numPatterns+1)]

#Values for all hyperparameters
hp = {'lrn_rate': 0.1,
      'epochs': 10,
      'lambdas': lambda_x,
      'val_mul': 10,
      'hidden': 15,
      'dataset': 'lagom'}

#Toggle which plots you want to see
lambda_loss_plot = False ; lambda_acc_plot = False

# For inputting seed from the command line.
#(i.e. use ```nohup nice n -19 python3 optimize_patterns.py seed &```,
#where seed is an integer).

try:
  seed = int(sys.argv[1])
except TypeError:
  warnings.warn("Seed input was not an integer. Using random seed instead.")
  seed = rand.randint(100,999)
except:
  warnings.warn("Seed input was missing. Using random seed instead.")
  seed = rand.randint(100,999)

#For hyperparameters in .txt
hyperparameter_string = str(list(hp.values()))
identifier = best_hash(hyperparameter_string)
uniqueFileName = "patterns_lambda_" + str(identifier) + "_" + str(seed) + ".txt"

tried_values = write_hyperparameters(uniqueFileName,hp,identifier,seed)

#This is for the final best lambda vs number of pattern plot
nPatterns = []
bestLambd = []

for number in extra_patterns: #range for the numbers of patterns
    # Create random number generators:
    # seed == -1 for random rng, seed >= 0 for fixed rng
    # (seed should be integer)
    # Reset the RNG seed
    data_seed = seed

    def generate_rng(seed):
        # seed should be an integer
        # -1 is for random rng, integer >= 0 for fixed
        if (seed != -1):
            return np.random.default_rng(data_seed)
        else:
            raise Exception('Please set a fixed seed, otherwise results will not be reproducable!')
            
    data_rng = generate_rng(data_seed)

    # Import data
    trn, val = sdg.generate_datasets(hp["dataset"],
                                 extra = number,
                                 val_mul = hp["val_mul"],
                                 try_plot = False,
                                 rng = data_rng)
    input_dim = len(trn[0][0]) #Get the input dimension from the training data
    
    #If the current number of patterns has already been tried with the same
    #settings, skip and move on to the next one
    if len(trn[0]) in tried_values:
        continue
    
    trainy = []
    valy = []
    lambd_to_cms = {}

    for lambd in lambda_x:
        #Reset the seed
        ann_seed = data_seed
        ann_rng = generate_rng(ann_seed)
        
        x_trn = trn[0]
        d_trn = trn[1]
    
        x_val = val[0]
        d_val = val[1]
    
        #Properties of all the layers
        #Recipe for defining a layer: [number of nodes, activation function, L2]
        layer_defines = [[hp["hidden"], act.tanh, lambd],
                         [1, act.sig, lambd]]
       
        #Create the model!
        test = ann.Model(input_dim, layer_defines, ann_rng)
        
        #Train the model!
        test.train(trn,val,hp["lrn_rate"],hp["epochs"]) #training, validation, lrn_rate, epochs, minibatchsize=0
        
        #Stuff for getting the confusion matrix, I think /Oskar
        outputs = test.feed_all_patterns(x_trn) #Collect the outputs for all the inputs
        cm_trn = cs.construct_confusion_matrix(outputs, d_trn)
    
        outputs = test.feed_all_patterns(x_val) #Collect the outputs for all the inputs
        cm_val = cs.construct_confusion_matrix(outputs, d_val)
    
        #For the loss vs lambda plot
        trainy.append(test.history['trn'][-1]) #Loss for validation
        valy.append(test.history['val'][-1]) #Loss for training
        
        #Not sure what this is doing????  /Oskar
        lambd_to_cms[lambd] = {"trn": cm_trn, "val": cm_val}
        

    #Appending the number of patterns nodes and best lambda for that number of
    #nodes, to be plotted later
    nPatterns.append(len(trn[0]))
    bestLambd.append(lambda_x[valy.index(min(valy))]) 
    
    #Write the "best lambda ; number of patterns" pairs
    #In the case of the test crashing, we should adapt the testing loop to
    #start off from where the previous highest number of patterns was
    with open(uniqueFileName, "a") as f:
        f.write("$ " + str(lambda_x[valy.index(min(valy))]) + " ; " + str(len(trn[0])) + "\n")
    
    if lambda_loss_plot:
    #The loss for each lambda is plotted once for every new number of patterns
        plt.figure()
        plt.plot(lambda_x, trainy, 'go', label='error after train vs lambd')
        plt.plot(lambda_x, valy, 'bo', label='Validation error vs lambd')
        plt.xlabel('lambdas')
        plt.ylabel('Error')
        plt.title(f'Error over lambdas for {extra_patterns} extra patterns')
        plt.legend()
        plt.savefig(f'error_lambda_plot_{extra_patterns}_extra_patterns.png')
        plt.show()
        plt.clf()
    
    # Construct a list of training and validation accuracies from the lambd_to_cms dictionary
    acc_trn = []
    acc_val = []
    for lambd in lambda_x:
        stats_trn = cs.calculate_stats(lambd_to_cms[lambd]["trn"])
        acc_trn.append(stats_trn[0]["advanced"]["acc"])
        stats_val = cs.calculate_stats(lambd_to_cms[lambd]["val"])
        acc_val.append(stats_val[0]["advanced"]["acc"])
    
    if lambda_acc_plot:
        # Plot the accuracy over lambda plot
        plt.figure()
        plt.plot(lambda_x, acc_trn, "go", label="Training")
        plt.plot(lambda_x, acc_val, "bo", label="Validation")
        plt.xlabel("$\lambda$")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over " + "$\lambda$" + " for " + str(extra_patterns) + " extra patterns")
        plt.legend()
        plt.savefig(f'accuracy_lambda_plot_{extra_patterns}_extra_patterns.png')
        plt.show()
        plt.clf()

