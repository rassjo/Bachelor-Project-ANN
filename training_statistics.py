"""Training Statistics
This script calculates and displays the training statistics by comparing the
network's output to the desired targets.

Call the stats_class() function with the appropriate parameters to
quickly get up and running.

Call construct_confusion_matrix() if you want to construct confusion matrices exclusively.
calculate_stats(), or calculate_and_display_stats() can then be called as desired,
taking the confusion matrix as arugment.

Useful resources:
    https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826     

WARNING:
    Recent changes made to the code have not yet been tested or re-documented.

WARNING:
    IF PANDAS ISN'T INSTALLED, THEN JUST COMMENT OUT THE SINGLE LINE THAT USES
    PANDAS DATAFRAME BEFORE PRINTING THE CONFUSION MATRIX.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_fundamental_class_metrics(confusion_matrix, class_num):
    """
    Get the fundamental class metrics from the confusion matrix.
    Parameters
    ----------
    confusion_matrix : nested np array
        The confusion matrix from which the metrics are to be determined.
    class_num : int
        The current class index.
    Returns
    -------
    tp : float
        Number of true positivies; Correct predictions of this class.
    tn : float
        Number of true negatives; Incorrect predictions of another class, when
        this class was correct.
    fp : float
        Number of false positives; Incorrect predictions of this class, when
        another class was correct.
    fn : float
        Number of false negatives; Predictions of another class, when it was
        some other class.
    """
    i = class_num # Set the current class index

    # True positive is [i, i]
    tp = np.sum(confusion_matrix[i, i])
    
    # True negative is not same row and not same column as i
    tn = np.sum(confusion_matrix[i+1:, i+1:])
    tn += np.sum(confusion_matrix[:i, :i])
    tn += np.sum(confusion_matrix[i+1:, :i])
    tn += np.sum(confusion_matrix[:i, i+1:])
    
    # False positive is same row as i
    fp = np.sum(confusion_matrix[i, i+1:])
    fp += np.sum(confusion_matrix[i, :i])
    
    # False negative is same column as i
    fn = np.sum(confusion_matrix[:i, i])
    fn += np.sum(confusion_matrix[i+1:, i])
    
    fundamental_class_metrics = {'tp': tp,
                                'tn': tn,
                                'fp': fp,
                                'fn': fn}
    
    return fundamental_class_metrics

def get_advanced_class_metrics(fundamental_class_metrics):
    """
    Calculate the class statistics from the fundamental class metrics.
    Parameters
    ----------
    fundamental_class_metrics : np array of floats
        The funamental class metrics; the number of true positives, true
        negatives, false positives and false negatives for this class.
    Returns
    -------
    acc : float
        Accuracy is the fraction of correct predictions.
    sens : float
        Sensitivity (recall) is the fraction of actual positivies that
        were correctly predicted.
    spec : float
        Specificity is the fraction of actual negatives that were
        correctly predicted.
    odds : float or str
        The odds-ratio is the “odds” you get a positive right divided
        by the odds you get a negative wrong. May have div by zero.
    prec : float
        Precision is the fraction of correct positives.
    f1 : float
        The F1-score is good for imbalanced classes, taking the
        harmonic mean of precision and sensitivity (recall).
    """
    # Unload the dictionary
    tp, tn, fp, fn = fundamental_class_metrics['tp'],
    fundamental_class_metrics['tn'],
    fundamental_class_metrics['fp'],
    fundamental_class_metrics['fn']
    
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
    
    advanced_class_metrics = {'acc': acc,
                             'sens': sens,
                             'spec': spec,
                             'odds': odds,
                             'prec': prec,
                             'f1': f1}
    
    return advanced_class_metrics


def print_fundamental_class_metrics(fundamental_class_metrics, indents = 1):
    """
    Print the provided fundamental metrics, with indentations.
    Parameters
    ----------
    class_fundamental_metrics : np array of floats
        The fundamental metrics to be printed.
    indents : int, optional
        The number of indentations to use on the section title, an extra
        indentation is used on the section contents. The default is 1.
    Returns
    -------
    None.
    """
    # Unload the dictionary
    tp, tn, fp, fn = fundamental_class_metrics['tp'],
    fundamental_class_metrics['tn'],
    fundamental_class_metrics['fp'],
    fundamental_class_metrics['fn']
    
    print('\t'*indents+'Fundamental metrics:')
    print('\t'*(indents+1)+'True positives: ' + str(tp) + ",")
    print('\t'*(indents+1)+'True negatives: ' + str(tn) + ",")
    print('\t'*(indents+1)+'False positives: ' + str(fp) + ",")
    print('\t'*(indents+1)+'False negatives: ' + str(fn) + ".")


def print_advanced_class_metrics(advanced_class_metrics, indents = 1):
    """
    Print the provided statistics, with indentations.
    Parameters
    ----------
    class_calculated_metrics : np array of floats
        The statistics to be printed.
    indents : int, optional
        The number of indentations to use on the section title, an extra
        indentation is used on the section contents. The default is 1.
    Returns
    -------
    None.
    """
    # Unload the dictionary
    acc, sens, spec, odds, prec, f1 = advanced_class_metrics['acc'],
    advanced_class_metrics['sens'],
    advanced_class_metrics['spec'],
    advanced_class_metrics['odds'],
    advanced_class_metrics['prec'],
    advanced_class_metrics['f1']
    
    print('\t'*indents+'Calculated metrics:')
    print('\t'*(indents+1)+'Accuracy = ' + str(acc) + ",")
    print('\t'*(indents+1)+'Sensitivity (recall) = ' + str(sens) + ",")
    print('\t'*(indents+1)+'Specificity = ' + str(spec) + ",")
    print('\t'*(indents+1)+'Odds ratio = ' +
          str(odds if odds != -1 else "div by zero") + ",")
    print('\t'*(indents+1)+'Precision = ' + str(prec) + ",")
    print('\t'*(indents+1)+'F1-score = ' + str(f1) + ".")

    
def heat_map(data, title = "Heatmap", xlabel = "X", ylabel = "Y"):
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
    plt.imshow(data, cmap='gray', interpolation='none')
    color_map_max = np.sum(data) / num_classes
    plt.clim(0, color_map_max) # Define the range of the colormap
    plt.title(title)
    plt.xticks(np.arange(num_classes))
    plt.yticks(np.arange(num_classes))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Write the value of each cell inside each respective cell
    for i in range(num_classes):
        for j in range(num_classes):
            contrasting_colour = (1, 1, 1)
            if (data[i, j] > color_map_max / 2):
                contrasting_colour = (0, 0, 0)
            plt.text(j, i, data[i, j], ha="center", va="center", color=contrasting_colour)        

    
def construct_confusion_matrix(outputs, targets):
    """
    Construct the confusion matrix from outputs and targets. Also contains and
    uses functions for converting float outputs into integer outputs or one-hot
    encoded arrays of integers.
    Parameters
    ----------
    outputs : numpy array of floats or nested numpy array of floats
        The predicted outputs of the ANN.
    targets : numpy array of floats or nested numpy array of floats
        The targets corresponding to the outputs.
    num_classes : int
        The total number of classes.
    is_binary : bool
        Whether or not it is a binary classification problem.
    Returns
    -------
    nested numpy array of ints
        The constructed confusion matrix.
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
    
    # Define oft used properties
    num_classes = targets.shape[-1]
    is_binary = num_classes == 2
    
    # Prepare the confusion matrix
    confusion_matrix = np.zeros(shape=(num_classes, num_classes), dtype=int)

    # Construct the confusion matrix
    for n in range(0, len(outputs)):
        output = outputs[n]
        target = targets[n]
        
        if (is_binary):
            output = binary(output)      
            confusion_matrix[output, target] += 1           
        else:
            output = multi(output)        
            confusion_matrix[output.index(1), target.index(1)] += 1
            
    return confusion_matrix

def calculate_stats(confusion_matrix):
    # Define oft used properties
    num_classes = len(confusion_matrix)
    
    # Prepare arrays for metrics
    fundamental_metrics = np.zeros(num_classes, dtype = int) # tp, tn, fp,
                                                             # fn for each class
    advanced_metrics = np.zeros(num_classes) # acc, sens (recall), spec,
                                             # odds, prec, f1 for each
                                             # class
    
    # Calculate the metrics and statistics using the confusion matrix
    for i in range(0, num_classes):
        fundamental_metrics[i] = get_fundamental_class_metrics(
            confusion_matrix, i)     
        advanced_metrics[i] = get_advanced_class_metrics(
            fundamental_metrics[i])
    
    statistics = {'fundamental': fundamental_metrics,
                 'advanced:' advanced_metrics}
    
    return statistics

def display_stats(confusion_matrix, statistics,
                  data_name, should_plot_cm):
    # Define oft used properties
    num_classes = len(confusion_matrix)
    is_binary = num_classes == 2
    
    # Print statistics
    print('\nStatistics for ' + data_name + ' data:')  
    
    # Plot or print confusion matrix
    if (should_plot_cm):
        # Plot confusion matrix
        heatmap(confusion_matrix, title = "Confusion matrix " + data_name,
                xlabel = "Targets", ylabel = "Outputs")
    else:
        # Print confusion matrix
        confusion_matrix_df = pd.DataFrame(confusion_matrix,
                                           range(num_classes),
                                           range(num_classes))   
        print("\tConfusion matrix: ")
        print(confusion_matrix_df)
    
    # Print metrics
    indents = 1 # Number of indents, for styling (multi-class requires
                # extra, as class name is a header)
    if (is_binary == False):
        indents = 2
    for i in range(0, num_classes):
        if (is_binary == False):
            print('\tClass ' + str(i) + " :")
        
        fundamental_class_metrics = statistics['fundamental'][i]
        advanced_class_metrics = statistics['advanced'][i]
        
        print_fundamental_class_metrics(fundamental_class_metrics, indents)
        print_advanced_class_metrics(advanced_class_metrics, indents)
        
        if (is_binary):
            break

def calculate_and_display_stats(confusion_matrix, data_name, should_plot_cm = True):  
    # Calculate stats (in: confusion matrix. out: fundamental metrics, advanced metrics)
    statistics = calculate_stats(confusion_matrix)
    
    # Display stats (in: stats. out: none (print and or plot))
    display_stats(confusion_matrix, statistics, data_name
                  should_plot_cm)
        
    return statistics

def class_stats(outputs, targets, data_name = '<data-name placeholder>',
                should_plot_cm = False):
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
    np array of floats or nested np array of floats
        for each class, an np array of
            acc : float
                Accuracy is the fraction of correct predictions.
            sens : float
                Sensitivity (recall) is the fraction of actual positivies that
                were correctly predicted.
            spec : float
                Specificity is the fraction of actual negatives that were
                correctly predicted.
            odds : float or str
                The odds-ratio is the “odds” you get a positive right divided
                by the odds you get a negative wrong. May have div by zero.
            prec : float
                Precision is the fraction of correct positives.
            f1 : float
                The F1-score is good for imbalanced classes, taking the
                harmonic mean of precision and sensitivity.
    """         
    # Construct confusion matrix (in: outputs, targets. out: confusion matrix)
    confusion_matrix = construct_confusion_matrix(outputs, targets)
    
    # Calculate and display stats (in: confusion matrix. out: fundamental metrics, advanced metrics (also print and or plot))
    statistics = calculate_and_display_stats(confusion_matrix, data_name, should_plot_cm)
    
    return statistics
