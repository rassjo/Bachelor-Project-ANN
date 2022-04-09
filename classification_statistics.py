"""Classification Statistics
This script calculates and displays the training statistics for
classification tasks by comparing the network's output to the desired targets.

Call the stats() function with the appropriate parameters to quickly
get up and running.

Call construct_confusion_matrix() if you want to construct confusion matrices
exclusively.
calculate_stats(), or calculate_and_display_stats() can then be called as
desired, taking the confusion matrix as argument.

Useful resources:
    https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826

WARNING:
    IF PANDAS ISN'T INSTALLED, THEN JUST COMMENT OUT THE SINGLE LINE THAT USES
    PANDAS DATA-FRAME BEFORE PRINTING THE CONFUSION MATRIX.

WARNING:
    THIS HAS NOT BEEN TESTED ON MULTI-CLASS CLASSIFICATION DATA.
"""
from pickle import NONE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ANN as ann
from matplotlib import colors

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
    dictionary of
        'tp' : float
            Number of true positives; Correct predictions of this class.
        'tn' : float
            Number of true negatives; Incorrect predictions of another class,
            when this class was correct.
        'fp' : float
            Number of false positives; Incorrect predictions of this class,
            when another class was correct.
        'fn' : float
            Number of false negatives; Predictions of another class, when it
            was some other class.
    """
    i = class_num  # Set the current class index

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
    fundamental_class_metrics : dictionary of str keys and int values
        The fundamental class metrics; the number of true positives,
        true negatives, false positives and false negatives for this class.
    Returns
    -------
    dictionary of
        'acc' : float
            Accuracy is the fraction of correct predictions.
        'sens' : float
            Sensitivity (recall) is the fraction of actual positives that
            were correctly predicted.
        'spec' : float
            Specificity is the fraction of actual negatives that were
            correctly predicted.
        'odds' : float or str
            The odds-ratio is the “odds” you get a positive right divided by
            the odds you get a negative wrong. May have div by zero.
        'prec' : float
            Precision is the fraction of correct positives.
        'f1' : float
            The F1-score is good for imbalanced classes, taking the harmonic
            mean of precision and sensitivity (recall).
    """
    # Unload the dictionary
    tp = fundamental_class_metrics['tp']
    tn = fundamental_class_metrics['tn']
    fp = fundamental_class_metrics['fp']
    fn = fundamental_class_metrics['fn']

    # Calculate the advanced metrics
    acc = -1  # In case of division by zero
    denominator = tp + fn + tn + fp
    if denominator != 0:
        acc = (tp + tn) / denominator
    sens = -1
    denominator = tp + fn
    if denominator != 0:
        sens = tp / denominator
    spec = -1
    denominator = tn + fp
    if denominator != 0:
        spec = tn / denominator
    odds = -1
    denominator = fn * fp
    if denominator != 0:
        odds = (tp * tn) / denominator
    prec = -1
    denominator = tp + fp
    if denominator != 0:
        prec = tp / denominator
    f1 = -1
    denominator = tp + 0.5*(fp + fn)
    if denominator != 0:
        f1 = tp / denominator
    
    advanced_class_metrics = {'acc': acc,
                              'sens': sens,
                              'spec': spec,
                              'odds': odds,
                              'prec': prec,
                              'f1': f1}
    
    return advanced_class_metrics


def print_fundamental_class_metrics(fundamental_class_metrics, indents=1):
    """
    Print the provided fundamental metrics, with indentations.
    Parameters
    ----------
    fundamental_class_metrics : dictionary of str keys and int values
        The fundamental metrics to be printed.
    indents : int, optional
        The number of indentations to use on the section title, an extra
        indentation is used on the section contents. The default is 1.
    Returns
    -------
    None.
    """
    # Unload the dictionary
    tp = fundamental_class_metrics.get('tp')
    tn = fundamental_class_metrics['tn']
    fp = fundamental_class_metrics['fp']
    fn = fundamental_class_metrics['fn']
    
    print('\t'*indents+'Fundamental metrics:')
    print('\t'*(indents+1)+'True positives: ' + str(tp) + ",")
    print('\t'*(indents+1)+'True negatives: ' + str(tn) + ",")
    print('\t'*(indents+1)+'False positives: ' + str(fp) + ",")
    print('\t'*(indents+1)+'False negatives: ' + str(fn) + ".")


def print_advanced_class_metrics(advanced_class_metrics, indents=1):
    """
    Print the provided statistics, with indentations.
    Parameters
    ----------
    advanced_class_metrics : dictionary of str keys and int values
        The statistics to be printed.
    indents : int, optional
        The number of indentations to use on the section title, an extra
        indentation is used on the section contents. The default is 1.
    Returns
    -------
    None.
    """
    # Unload the dictionary
    acc = advanced_class_metrics['acc']
    sens = advanced_class_metrics['sens']
    spec = advanced_class_metrics['spec']
    odds = advanced_class_metrics['odds']
    prec = advanced_class_metrics['prec']
    f1 = advanced_class_metrics['f1']
    
    print('\t'*indents+'Calculated metrics:')
    print('\t'*(indents+1)+'Accuracy = ' +
          (str(acc) if acc != -1 else "div by zero") + ",")
    print('\t'*(indents+1)+'Sensitivity (recall) = ' +
          (str(sens) if sens != -1 else "div by zero") + ",")
    print('\t'*(indents+1)+'Specificity = ' +
          (str(spec) if spec != -1 else "div by zero") + ",")
    print('\t'*(indents+1)+'Odds ratio = ' +
          (str(odds) if odds != -1 else "div by zero") + ",")
    print('\t'*(indents+1)+'Precision = ' +
          (str(prec) if prec != -1 else "div by zero") + ",")
    print('\t'*(indents+1)+'F1-score = ' +
          (str(f1) if f1 != -1 else "div by zero") + ".")


def heat_map(data, title="<title placeholder>", x_label="<x_label placeholder>",
             y_label="<y_label placeholder>"):
    """
    Plots heat map figure and color bar.
    Parameters
    ----------
    data : nested numpy array of ints
        The grid data to be plotted as a heat map
    title : str, optional
        Title of the figure. The default is "<title placeholder>".
    x_label : str, optional
        x-axis label. The default is "<x_label placeholder>".
    y_label : str, optional
        y-axis label. The default is "<y_label placeholder>".
    Returns
    -------
    None.
    """
    # Define oft used properties
    num_classes = len(data)

    # Create figure and plot the data
    fig, ax1 = plt.subplots()
    im = ax1.imshow(data, cmap='gray', interpolation='none')

    # Define the range of the colormap
    color_map_max = np.sum(data) / num_classes
    im.set_clim(0, color_map_max)

    # Write the title
    ax1.set_title(title)

    # Write the ticks
    ax1.set_xticks(np.arange(num_classes))
    ax1.set_yticks(np.arange(num_classes))

    # Write axis labels (uncomment line endings to move to top left)
    ax1.set_xlabel(x_label)  # , loc="left")
    ax1.set_ylabel(y_label)  # , loc="top")

    # Move x-axis label and ticks to the top
    ax1.xaxis.set_label_position("top")
    ax1.xaxis.tick_top()

    # Write the value of each cell inside each respective cell
    for i in range(num_classes):
        for j in range(num_classes):
            contrasting_colour = (1, 1, 1)
            if data[i, j] > color_map_max / 2:
                contrasting_colour = (0, 0, 0)
            ax1.text(j, i, data[i, j], ha="center", va="center",
                     color=contrasting_colour)

    # Fix padding and show
    fig.tight_layout()
    plt.show()

    
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
    Returns
    -------
    nested numpy array of ints
        The constructed confusion matrix.
    """
    def binary(value):
        """
        Convert floats in the range [0,1] to nearest integer.
    
        Parameters
        ----------
        value : float
            The un-rounded float output.
    
        Returns
        -------
        int
            The rounded output.
    
        """
        if value > 0.5:
            value = 1
        else:
            value = 0
        return value

    def multi(values):
        """
        Converts an output array of floats into a one-hot encoded output array
        of integers.
    
        Parameters
        ----------
        values : numpy array of floats
            The un-rounded array of floats.
    
        Returns
        -------
        numpy array of ints
            The one-hot encoded array of ints.
    
        """     
        hot_index = np.argmax(values)
        for i in range(0, len(values)):
            values[i] = 0
        values[hot_index] = 1
        return values
    
    # Define oft used properties
    if isinstance(targets.shape, tuple) and (len(targets.shape) == 1):
        num_classes = 2
    else:
        num_classes = targets.shape[-1]
    is_binary = num_classes == 2
    
    # Prepare the confusion matrix
    confusion_matrix = np.zeros(shape=(num_classes, num_classes), dtype=int)

    # Construct the confusion matrix
    for n in range(0, len(outputs)):
        output = outputs[n]
        target = targets[n]
        
        if is_binary:
            output = binary(output)      
            confusion_matrix[output, target] += 1           
        else:
            output = multi(output)        
            confusion_matrix[output.index(1), target.index(1)] += 1
            
    return confusion_matrix


def calculate_stats(confusion_matrix):
    """
    Calculates the fundamental and advanced metrics from a given confusion
    matrix.

    Parameters
    ----------
    confusion_matrix : nested numpy array of ints
        The confusion matrix from which statistics are to be determined.

    Returns
    -------
    tuple of tuples of dictionaries
        The fundamental and advanced metrics.

    """
    # Define oft used properties
    num_classes = len(confusion_matrix)

    # Prepare list for metrics
    statistics = np.ndarray(num_classes, dtype=dict)

    # Calculate the metrics and statistics using the confusion matrix
    for i in range(0, num_classes):
        fundamental_metrics = get_fundamental_class_metrics(confusion_matrix, i)
        advanced_metrics = get_advanced_class_metrics(fundamental_metrics)
        statistics[i] = ({'fundamental': fundamental_metrics,
                          'advanced': advanced_metrics})

    # Convert statistics list to (an immutable) tuple and return
    return tuple(statistics)


def display_stats(confusion_matrix, statistics, data_name, should_plot_cm=True):
    """
    Display the fundamental and advanced metrics from a given confusion
    matrix.

    Parameters
    ----------
    confusion_matrix : nested numpy array of ints
        The confusion matrix from which statistics are to be determined.
        Horizontally are the targets, vertically is predicted.
    statistics : tuple of tuples of dictionaries
        The fundamental and advanced metrics.
    data_name : str
        The name of the data to be printed with the statistics and used in
        the plot title.
    should_plot_cm : bool
        Whether or not the confusion matrix should be plotted. Otherwise,
        the confusion matrix will be printed.

    Returns
    -------
    None.
    """
    # Define oft used properties
    num_classes = len(confusion_matrix)
    is_binary = num_classes == 2
    
    # Print statistics
    print('\nStatistics for ' + data_name + ' data:')

    # Plot or print confusion matrix
    if should_plot_cm:
        # Plot confusion matrix
        heat_map(confusion_matrix, title="Confusion matrix " + data_name,
                 x_label="Target class", y_label="Output class")
    else:
        # Print confusion matrix
        confusion_matrix_df = pd.DataFrame(confusion_matrix,
                                           range(num_classes),
                                           range(num_classes))   
        print("\t" + "Confusion matrix: ")
        print("\t"*2 + "Horizontally: Targets. Vertically: Outputs.")
        print(confusion_matrix_df)
    
    # Print metrics
    indents = 1  # Number of indents, for styling (multi-class requires
    # extra, as class name is a header)
    if not is_binary:
        indents = 2
    for i in range(0, num_classes):
        if not is_binary:
            print('\tClass ' + str(i) + " :")

        fundamental_class_metrics = statistics[i]['fundamental']
        advanced_class_metrics = statistics[i]['advanced']
        print_fundamental_class_metrics(fundamental_class_metrics, indents)
        print_advanced_class_metrics(advanced_class_metrics, indents)
        
        if is_binary:
            break


def calculate_and_display_stats(confusion_matrix,
                                data_name='<data_name placeholder>',
                                should_plot_cm=True):
    """
    Calculates and prints the statistics from the predicted outputs and true
    targets of classification tasks.

    Parameters
    ----------
    confusion_matrix : nested numpy array of ints
        The confusion matrix from which statistics are to be determined.
        Horizontally are the targets, vertically is predicted.
    data_name : str, optional
        The name of the data to be printed. The default is
        '<data_name placeholder>'.
    should_plot_cm : bool, optional
        Whether or not the confusion matrix should be plotted. If False, then
        it is printed instead. The default is True.
    Returns
    -------
    tuple of tuples of dictionaries
        The fundamental and advanced metrics.
    """

    # Calculate stats (in: confusion matrix. out: fundamental metrics,
    # advanced metrics)
    statistics = calculate_stats(confusion_matrix)

    # Display stats (in: stats. out: none (print and or plot))
    display_stats(confusion_matrix, statistics, data_name, should_plot_cm)
        
    return statistics


def stats(outputs, targets, data_name='<data_name placeholder>',
          should_plot_cm=True):
    """
    Calculates and prints the statistics from the predicted outputs and true
    targets of classification tasks.

    Parameters
    ----------
    outputs : numpy array of floats or nested numpy array of ints
        The predicted outputs of the neural network.
    targets : numpy array of floats or nested numpy array of ints
        The desired targets of the classification task.
    data_name : str, optional
        The name of the data to be printed.
        The default is '<data_name placeholder>'.
    should_plot_cm : bool, optional
        Whether or not the confusion matrix should be plotted. If False, then
        it is printed instead. The default is True.
    Returns
    -------
    tuple of tuples of dictionaries
        The fundamental and advanced metrics.
    """         
    # Construct confusion matrix (in: outputs, targets. out: confusion matrix)
    confusion_matrix = construct_confusion_matrix(outputs, targets)
    
    # Calculate and display stats (in: confusion matrix. out: fundamental
    # metrics, advanced metrics (also print and or plot))
    statistics = calculate_and_display_stats(confusion_matrix, data_name,
                                             should_plot_cm)
    
    return statistics

def decision_boundary_1d(patterns, targets, model, precision = 0.025):
    """
    For binary classification problems only.
    Precision is the grid step-size from which the decision boundary is created.
    """

    input_dim = len(patterns[0])
    if (input_dim != 1): # Only plot if it is a 1-dimensional problem
        return None

    x_min, x_max = patterns[:, 0].min() - .25, patterns[:, 0].max() + .25
    #y_min, y_max = patterns[:, 1].min() - .5, patterns[:, 1].max() + .5

    #x_min, x_max = -3.7, 2.1
    y_min, y_max = -0.6, 0.9

    #xx, yy = np.meshgrid(np.arange(x_min, x_max, precision), np.arange(y_min, y_max, precision)) #x_max*1.1
    #Z = model.feed_all_patterns(np.c_[xx.ravel(), yy.ravel()])
    xx = [[x_min]]
    x = x_min
    while x < x_max:
        x += precision
        xx.append([x])

    xx_plain = [x_min]
    x = x_min
    while x < x_max:
        x += precision
        xx_plain.append(x)

    yy = [y_min]
    y = y_min
    while y < y_max:
        y += precision
        yy.append(y)

    xx = np.array(xx)
    yy = np.array(yy)

    #print(xx)

    Z = model.feed_all_patterns(xx)

    Z_new = []
    for i in range(0, len(yy)):
        Z_new.append(Z)

    Z_new = np.array(Z_new)
    
    Z[Z>.5] = 1
    Z[Z<= .5] = 0

    Y_pr = model.feed_all_patterns(patterns).reshape(targets.shape)
  
    Y = np.copy(targets)
    Y_pr[Y_pr>.5] = 1
    Y_pr[Y_pr<= .5] = 0
    Y[(Y!=Y_pr) & (Y==0)] = 2
    Y[(Y!=Y_pr) & (Y==1)] = 3
    
    plt.figure()

    # Create countor line
    # make a color map of fixed colors
    cmap = colors.ListedColormap(['blue', 'red'])
    bounds=[0, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    #colour_map = 'bwr' #'brg'
    plt.contourf(xx_plain, yy, Z_new, cmap=cmap, norm=norm, alpha = 0.33) # Region-coloured background
    #plt.contour(xx, yy, Z, cmap=plt.cm.Paired) # Transparent background
    
    # Markers: Circles 'o', squares 's', triangles '^'

    colour_1 = (1,0,0,1)
    colour_1_error = (1,0,0,1)
    colour_2 = (0,0,1,1)
    colour_2_error = (0,0,1,1)
    
    patterns = list(patterns)
    for i in range(0, len(patterns)):
        patterns[i] = list(patterns[i])
        patterns[i].append(0)
        patterns[i] = np.array(patterns[i])
    patterns = np.array(patterns)

    zeros_0 = []
    for i in range(0, len(patterns[:, 0][Y==0])):
        zeros_0.append(0)
    zeros_1 = []
    for i in range(0, len(patterns[:, 0][Y==1])):
        zeros_1.append(0)
    zeros_2 = []
    for i in range(0, len(patterns[:, 0][Y==2])):
        zeros_2.append(0)
    zeros_3 = []
    for i in range(0, len(patterns[:, 0][Y==3])):
        zeros_3.append(0)

    #ys = np.zeros(patterns[:, 0][Y==1])

    # Plot class A
    plt.scatter(patterns[:, 0][Y==0], zeros_0, marker='o', color=colour_2, edgecolors='none', label='Class A')
    plt.scatter(patterns[:, 0][Y==2], zeros_2, marker = 'o', color=colour_2_error, edgecolors='none')

    # Plot class B
    plt.scatter(patterns[:, 0][Y==1], zeros_1, marker='^', color=colour_1, edgecolors='none', label='Class B')
    plt.scatter(patterns[:, 0][Y==3], zeros_3, marker = '^', color=colour_1_error, edgecolors='none') 
    
    plt.xlabel('Input $x$')
    #plt.ylabel('Input $y$')

    # Make x and y axis scale equally
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.yticks([])
                                                   
    plt.legend()

    plt.draw()


def decision_boundary(patterns, targets, model, precision = 0.025):
    """
    For binary classification problems only.
    Precision is the grid step-size from which the decision boundary is created.
    """
    input_dim = len(patterns[0])
    if (input_dim != 2): # Only plot if it is a 2-dimensional problem
        return None

    x_min, x_max = patterns[:, 0].min() - .25, patterns[:, 0].max() + .25
    y_min, y_max = patterns[:, 1].min() - .25, patterns[:, 1].max() + .25

    #x_min, x_max = -3.7, 2.1
    #y_min, y_max = -5, 5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, precision), np.arange(y_min, y_max, precision)) #x_max*1.1
    Z = model.feed_all_patterns(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    Z[Z>.5] = 1
    Z[Z<= .5] = 0

    Y_pr = model.feed_all_patterns(patterns).reshape(targets.shape)
  
    Y = np.copy(targets)
    Y_pr[Y_pr>.5] = 1
    Y_pr[Y_pr<= .5] = 0
    Y[(Y!=Y_pr) & (Y==0)] = 2
    Y[(Y!=Y_pr) & (Y==1)] = 3
    
    plt.figure()

    # Create countor line
    # make a color map of fixed colors
    cmap = colors.ListedColormap(['blue', 'red'])
    bounds=[0, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    #colour_map = 'bwr' #'brg'
    plt.contourf(xx, yy, Z, cmap=cmap, norm=norm, alpha = 0.33) # Region-coloured background
    #plt.contour(xx, yy, Z, cmap=plt.cm.Paired) # Transparent background
    
    # Markers: Circles 'o', squares 's', triangles '^'

    colour_1 = (1,0,0,1)
    colour_1_error = (1,0,0,1)
    colour_2 = (0,0,1,1)
    colour_2_error = (0,0,1,1)
    
    # Plot class A
    plt.scatter(patterns[:, 0][Y==0], patterns[:, 1][Y==0], marker='o', color=colour_2, edgecolors='none', label='Class A')
    plt.scatter(patterns[:, 0][Y==2], patterns[:, 1][Y==2], marker = 'o', color=colour_2_error, edgecolors='none')

    # Plot class B
    plt.scatter(patterns[:, 0][Y==1], patterns[:, 1][Y==1], marker='^', color=colour_1, edgecolors='none', label='Class B')
    plt.scatter(patterns[:, 0][Y==3], patterns[:, 1][Y==3], marker = '^', color=colour_1_error, edgecolors='none')  
    
    plt.xlabel('Input $x$')
    plt.ylabel('Input $y$')

    # Make x and y axis scale equally
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
                                                   
    plt.legend()

    plt.draw()
