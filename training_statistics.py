"""Training Statistics

This script calculates and displays the training statistics by comparing the
network's output to the desired targets.
Call the stats_class function with the appropriate parameters to
quickly get up and running.

TO DO: DOCUMENT AND COMMENT EACH FUNCTION
       ADD STATISTICS (i.e. accuracy, etc.) FOR MULTI-CLASSIFICATION,
       COULD THIS BE MADE MORE GENERIC?
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def heatmap(data, title = "Heatmap", xlabel = "X", ylabel = "Y"):
    num_classes = len(data)
    plt.figure(figsize = (num_classes, num_classes))
    plt.imshow(data)
    plt.title(title)
    plt.xticks(np.arange(num_classes))
    plt.yticks(np.arange(num_classes))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    # Annotate the data value into respective cell
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, data[i, j], ha="center", va="center", color="w")        

def stats_class(outputs, targets, data_name = 'training', should_plot_cm = False):
   
    def binary(output):
        if (output > 0.5):
            output = 1
        else:
            output = 0
        return(output)
    
    def multi(outputs): # convert multi-classification outputs to one-hot.
        hot_index = np.argmax(outputs)        
        for i in range(0, len(outputs)):
            outputs[i] = 0        
        outputs[hot_index] = 1
        return(outputs)          
    
    num_classes = len(outputs[0]) if len(outputs[0]) > 1 else 2    
    confusion_matrix = np.zeros(shape=(num_classes, num_classes), dtype=int)
    
    is_binary = num_classes == 2
    tp, tn, fp, fn = 0, 0, 0, 0
    N = len(outputs)
    for n in range(0, N):
        output = outputs[n]
        target = targets[n] 
        
        if (is_binary):
            output = output[0]
            output = binary(output)  
            
            confusion_matrix[output, target] += 1
            
            if (output > 0.5):
                if (output == target):
                    tp += 1
                else:
                    fp += 1
            else:
                if (output == target):
                    tn += 1
                else:
                    fn += 1
                    
        else:
            output = multi(output)        
            
            confusion_matrix[output.index(1), target.index(1)] += 1
    
    print('\nStatistics for ' + data_name + ' data:')
    if (is_binary):
        acc = (tp + tn) / (tp + fn + tn + fp) # fraction of correct predictions
        sens = tp / (tp + fn) # fraction of actual positivies that were correctly
                              # predicted
        spec = tn / (tn + fp) # fraction of actual negatives that were correctly
                              # predicted
        odds = "div by zero" # in case of division by zero
        if (fn * fp != 0):
            odds = (tp * tn) / (fn * fp) # the “odds” you get a positive right
                                         # divided by the odds you get a negative
                                         # wrong
        print('Accuracy = ' + str(acc))
        print('Sensitivity = ' + str(sens))
        print('Specificity = ' + str(spec))
        print('Odds ratio = ' + str(odds))
    
    if (should_plot_cm):
        heatmap(confusion_matrix, title = "Confusion matrix " + data_name, xlabel = "Outputs", ylabel = "Targets")
    else:
        confusion_matrix = pd.DataFrame(confusion_matrix, range(num_classes), range(num_classes))   
        print("Confusion matrix: ")
        print(confusion_matrix)

    return acc, sens, spec, odds
