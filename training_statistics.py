"""Training Statistics

This script calculates and displays the training statistics by comparing the
network's output to the desired targets.
Call the stats_class function with the appropriate parameters to
quickly get up and running.

Useful resources:
    https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826     

WARNING:
    IF PANDAS ISN'T INSTALLED, THEN JUST COMMENT OUT THE SINGLE LINE THAT USES
    PANDAS DATAFRAME BEFORE PRINTING THE CONFUSION MATRIX.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def heatmap(data, title = "Heatmap", xlabel = "X", ylabel = "Y"):
    """
    Plots heatmap figure and colorbar.

    Parameters
    ----------
    data : nested numpy array of ints
        The grid data to be plotted as a heatmap
    title : str, optional
        Title of the figure. The default is "Heatmap".
    xlabel : str, optional
        x-axis label. The default is "X".
    ylabel : str, optional
        y-axis label. The default is "Y".

    Returns
    -------
    None.

    """
    num_classes = len(data)
    plt.figure(figsize = (num_classes, num_classes))
    plt.imshow(data)
    plt.title(title)
    plt.xticks(np.arange(num_classes))
    plt.yticks(np.arange(num_classes))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    # Write the value of each cell inside each respective cell
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, data[i, j], ha="center", va="center", color="w")        

def class_stats(outputs, targets, data_name = 'training', should_plot_cm = False):
    """
    Calculates and prints the statistics from the predicted outputs and true
    targets of classification tasks.
    
    WARNING: THIS HAS NOT BEEN TESTED ON MULTI-CLASS CLASSIFICATION DATA.

    Parameters
    ----------
    outputs : numpy array of floats or nested numpy array of ints
        The predicted outputs of the neural network.
    targets : numpy array of floats or nested numpy array of ints
        The desired targets of the classification task.
    data_name : str, optional
        The name of the data to be printed. The default is 'training'.
    should_plot_cm : bool, optional
        Whether or not the confusion matrix should be plotted. If False, then
        it is printed instead. The default is False.

    Returns
    -------
    acc : float
        Accuracy is the fraction of correct predictions.
    sens : float
        Sensitivity is the fraction of actual positivies that were correctly
        predicted.
    spec : float
        Specificity is the fraction of actual negatives that were correctly
        predicted.
    odds : float
        The odds-ratio is the “odds” you get a positive right divided by the
        odds you get a negative wrong.

    """
    
    def binary(output):
        """
        Convert float binary outputs in the range [0,1] to nearest integer 0
        or 1.

        Parameters
        ----------
        output : float
            The unrounded float output.

        Returns
        -------
        int
            The rounded output.

        """
        if (output > 0.5):
            output = 1
        else:
            output = 0
        return(output)
    
    def multi(outputs): # convert multi-classification outputs to one-hot.
        """
        Converts an output array of floats into a one-hot encoded output array
        of integers.

        Parameters
        ----------
        outputs : numpy array of floats
            The unrounded array of floats.

        Returns
        -------
        numpy array of ints
            The one-hot encoded array of ints.

        """     
        hot_index = np.argmax(outputs)        
        for i in range(0, len(outputs)):
            outputs[i] = 0        
        outputs[hot_index] = 1
        return(outputs)          
    
    num_classes = len(outputs[0]) if len(outputs[0]) > 1 else 2    
    confusion_matrix = np.zeros(shape=(num_classes, num_classes), dtype=int)
    
    is_binary = num_classes == 2
    N = len(outputs)
    
    # Construct the confusion matrix
    for n in range(0, N):
        output = outputs[n]
        target = targets[n] 
        
        if (is_binary):
            output = output[0]
            output = binary(output)      
            confusion_matrix[output, target] += 1
                    
        else:
            output = multi(output)        
            confusion_matrix[output.index(1), target.index(1)] += 1
    
    fundamental_metrics = np.zeros((num_classes, 4), dtype = int) # tp, tn, fp,
                                                            # fn for each class
    calculated_metrics = np.zeros((num_classes, 6)) # acc, sens (recall), spec,
                                                    # odds, prec, f1 for each
                                                    # class
    
    # Calculate the metrics and statistics using the confusion matrix
    for i in range(0, num_classes):
        # True positive is [i, i]
        # True negative is not same row and not same column as i
        # False positive is same row as i
        # False negative is same column as i
        
        tp = np.sum(confusion_matrix[i, i])
        
        tn = np.sum(confusion_matrix[i+1:, i+1:])
        tn += np.sum(confusion_matrix[:i, :i])
        tn += np.sum(confusion_matrix[i+1:, :i])
        tn += np.sum(confusion_matrix[:i, i+1:])
        
        fp = np.sum(confusion_matrix[i, i+1:])
        fp += np.sum(confusion_matrix[i, :i])
        
        fn = np.sum(confusion_matrix[:i, i])
        fn += np.sum(confusion_matrix[i+1:, i])
        
        fundamental_metrics[i, :] = tp, tn, fp, fn
        
        acc = (tp + tn) / (tp + fn + tn + fp) # fraction of correct predictions
        sens = tp / (tp + fn) # fraction of actual positivies that were correctly
                              # predicted (same as recall)
        spec = tn / (tn + fp) # fraction of actual negatives that were correctly
                              # predicted
        odds = -1             # -1 in case of division by zero
        if (fn * fp != 0):
            odds = (tp * tn) / (fn * fp) # the “odds” you get a positive right
                                         # divided by the odds you get a negative
                                         # wrong
        prec = tp / (tp + fp)
        f1 = tp / (tp + 0.5*(fp + fn))  

        calculated_metrics[i, :] = acc, sens, spec, odds, prec, f1             
    
    # Print calculated statistics
    print('\nStatistics for ' + data_name + ' data:')    
    for i in range(0, num_classes):
        if (is_binary == False):
            print('\nClass ' + str(i) + " :")
        
        tp, tn, fp, fn = fundamental_metrics[i, :]
        acc, sens, spec, odds, prec, f1 = calculated_metrics[i, :]
        
        print('\nFunamental metrics:')
        print('True positives: ' + str(tp))
        print('True negatives: ' + str(tn))
        print('False positives: ' + str(fp))
        print('False negatives: ' + str(fn))
        
        print('\nCalculated metrics:')
        print('Accuracy = ' + str(acc))
        print('Sensitivity (recall) = ' + str(sens))
        print('Specificity = ' + str(spec))
        print('Odds ratio = ' + str(odds if odds != -1 else "div by zero"))
        print('Precision = ' + str(prec))
        print('F1-score = ' + str(f1))
        
        if (is_binary):
            break
    
    # Plot or print confusion matrix
    if (should_plot_cm):
        heatmap(confusion_matrix, title = "Confusion matrix " + data_name, xlabel = "Outputs", ylabel = "Targets")
    else:
        confusion_matrix = pd.DataFrame(confusion_matrix, range(num_classes), range(num_classes))   
        print("\nConfusion matrix: ")
        print(confusion_matrix)

    # Return calculated statistics
    return calculated_metrics
