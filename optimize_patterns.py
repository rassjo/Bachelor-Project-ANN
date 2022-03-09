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
number_to_try = 2

#For patterns
numPatterns = 5 #how many (extra) times we make a new number of patterns

#For the model
learnRate = 0.1
epochs = 10
minibatchsize = 0 #0 if we don't want to use minibatches

lambda_x = [0.00001*10**(int(0.25*i))*2**(i%4) for i in range(0,number_to_try)]
start_la = min(lambda_x)
final_la = max(lambda_x)

nPatterns = []
bestLambd = []
hid = 15 #hidden nodes
preset_name = "lagom"
val_mul = 10
for i in range(0,numPatterns+1): #range for the numbers of patterns
    trainy = []
    valy = []
    lambd_to_cms = {}
    extra_patterns = 4*10**(int(0.25*i))*2**(i%4) #patterns = constx10^(int(0.25xtrial))x2^(trial%4)
    
    for lambd in lambda_x: #number of lambdas we want to test for each number of hidden nodes
        #------------------------------------------------------------------------------
        # Create random number generators:
        # seed == -1 for random rng, seed >= 0 for fixed rng (seed should be integer)
        data_seed = 2
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
        trn, val = sdg.generate_datasets(preset_name,
                                     extra = extra_patterns,
                                     val_mul = val_mul,
                                     try_plot = False,
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
        # Save as string for .txt.
        # This is a disgusting way of getting the string... but it wasn't playing nice otherwise.
        layer_defines_str = "[[" + str(hid) +", act.tanh, " + str(lambd) + "], [1, act.sig, " + str(lambd) + "]]"
    
        test = ann.Model(input_dim, layer_defines, ann_rng)
    
        #Check results
        answer1 = check_results(test, False)
    
        outputs = test.feed_all_patterns(x_trn) # Collect the outputs for all the inputs
        outputs = test.feed_all_patterns(x_val) # Collect the outputs for all the inputs
    
        # plt.show()
        # plt.clf()
    
        test.train(trn,val,learnRate,epochs,minibatchsize) #training, validation, lrn_rate, epochs, minibatchsize=0
    
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
plt.figure()
plt.errorbar(nPatterns, bestLambd, yerr=0, fmt = 'o', color='r',  ecolor='b', capsize=5, label='best lambdas vs # patterns')
plt.xlabel('# patterns')
plt.ylabel('best lambdas')
plt.title('best lambdas vs # patterns')
plt.legend()
plt.savefig('patterns_lambda_plot.png')
plt.show()
plt.clf()

uniqueFileName = "patterns_lambda_for_data_seed_"+str(data_seed)+".txt"
def write_hyperparameters():
    # If the document already exists, then do nothing.
    try:
        open(uniqueFileName, "x")
    except:
        return None
    
    # Overwrite the empty file with the hyperparameters
    with open(uniqueFileName, "w") as f:
        # Write hyperparameters
        f.write("# Here we describe the hyperparameters")
        # Lambda stuff
        f.write("# Lambda stuff" + "\n")
        f.write(str("# number of lambdas ; start lambda ; final lambda" + "\n"))
        f.write(str(number_to_try) + " ; " + str(start_la) + " ; " + str(final_la) + "\n")
        f.write(str("\n"))
        # Pattern stuff
        f.write("# Pattern stuff" + "\n")
        f.write(str("# number of patterns" + "\n"))
        f.write(str(numPatterns) + "\n")
        f.write(str("\n"))
        # ANN stuff
        f.write("# ANN stuff" + "\n")
        f.write(str("# learning rate ; epochs ; mini-batch size ; layer defines (ignore lambda) ; ann seed" + "\n"))
        f.write(str(learnRate) + " ; " + str(epochs) + " ; " + str(minibatchsize) + " ; " + layer_defines_str + " ; " + str(ann_seed) + "\n")
        f.write(str("\n"))
        # Data-set stuff
        f.write("# Data-set stuff" + "\n")
        f.write(str("# preset name ; validation to training ratio ; data seed" + "\n"))
        f.write(str(preset_name) + " ; " + str(val_mul) + " ; " + str(data_seed) + "\n")
        f.write(str("\n"))

        # Write semi-colon seperated optimal lambda for number of patterns on each line
        f.write("# Here we write the results" + "\n")
        f.write(str("# optimal lambda ; number of patterns") + "\n")

# Ideally, we would call this before running everything...
# That way, we could then write the "best lambda ; number of patterns" as soon as they are available,
# rather than waiting for everything to finish.
# But at the moment, it relies on a few hyperparameters that we don't define until during testing.
write_hyperparameters()

# Write the "best lambda ; number of patterns" pairs
# In the case of the test crashing, we should adapt the testing loop to start off from where the
# previous highest number of patterns was
with open(uniqueFileName, "a") as f:
    for i in range(0, len(bestLambd)):
        f.write(str(bestLambd[i]) + " ; " + str(nPatterns[i]) + "\n")
