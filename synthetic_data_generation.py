"""Synthetic Data Generator

This script contains various procedures for data generation.
Call the generate_datasets function with the appropriate parameters to
quickly get up and running.

TO DO: ADD 'help', THAT DISPLAYS THE AVAILABLE PRESETS.
       CAN PLOTTING BE SIMPLIFIED BY USING NUMPY.T (TRANSPOSE), RATHER THAN
       CONVERING TO LISTS, USING zip, AND CONVERTING BACK?
"""
import numpy as np
import matplotlib.pyplot as plt

def load_presets(file_name):
    """
    Loads the presets from the specified preset file.

    Parameters
    ----------
    file_name : str
        Path to the text file to loads the presets from.

    Returns
    -------
    ...
        The properties defining each available preset.
    """
    datasets = {}
    
    with open(file_name, 'r') as defines:
    
        for line in defines:
                #Ignore comments in the text file
                if line[0] == "#" or line == '\n' or line == '':
                    continue
                #Remove the new line command
                line = line[:-1]
                #Split up all properties that are used to define the dataset
                properties = line.split(' ; ')
                #First is the name
                name = properties[0]
                #Then the input dimension
                dimensions = int(properties[1])
                #Iterate over all the distributions that we have defined members
                # for
                members = np.array([int(number) for number in properties[2].split('/')])
                #Split up the coordinates for the different distributions
                center_coordinates = properties[3].split('/') ; coordinates = []
                #Turn the coordinates from a list of strings to a list of lists of numbers
                for location in center_coordinates:
                    coordinates.append([float(number) for number in location[1:-1].split(',')])
                #Make the list an array
                centers = np.array(coordinates)
                #Do the same with the scales as the coordinates
                scale_values = properties[4].split('/') ; sizes = []
                for size in scale_values:
                    sizes.append([float(number) for number in size[1:-1].split(',')])
                scales = np.array(sizes)
                # Get the (optional) class identifiers.
                if len(properties) >= 6:
                    class_ids = properties[5].split('/')
                    class_ids = np.array(class_ids)
                else:
                    # Just use a new integer for each distribution, if no class
                    # identifiers were provided.
                    class_ids = np.arange(0, len(members))

                #Add the dataset to the dictionary
                datasets[name] = [dimensions, members, centers, scales,
                                  class_ids]
    
    return datasets


def generate_class_data(num_dims, mems, centers, scales, class_ids, val_mul=1,
                        rng=np.random.default_rng()):
    """
    Generates data for a specified-dimensional multi-class classification
    problem from parameter-specified normal distributions.

    Parameters
    ----------
    num_dims : int
        The number of dimensions for the data to be generated in.
    mems : numpy array of integers
        The number of members of each class.
    centers : nested numpy array of floats
        The num_dims-dimensional displacements of each class.
    scales : nested numpy array of floats
        The num_dims-dimensional standard-deviation (scale) of each class.
    class_ids : numpy array
        The class ids of the distributions. Use the same ids to make multiple
        distributions part of the same class.
    val_mul : int
        The relative size (in members) of the validation data-set, with respect
        to the (training data-set). Set val_mul = 0 for no validation data-set.
    rng : numpy generator
        The random number generator. Default is random unless a rng with a fixed
        seed is provided.

    Returns
    -------
    nested numpy array of floats
        The generated numpy array of I-dimensional co-ordinates.
    numpy array of ints or nested numpy array of ints
        The classifications for each data-point. For binary classifications,
        each target is an integer 0 or 1. For multi-class classification,
        the target uses one-hot encoding.
    """
    if (val_mul == 0):
        return (None, None)
    
    #We have to define a new variable explicitly, otherwise if we just
    #modify the argument that we sent to the function we actually modify
    #the original array and the changes remain after we leave the function
    #because arrays are fucking stupid.
    num_mems = mems*val_mul

    num_distributions = len(class_ids)
    unique_classes = set(class_ids)
    num_classes = len(unique_classes)
      
    x = np.empty(shape=(int(np.sum(num_mems)), num_dims), dtype=np.float32)
    d = np.zeros(shape=(int(np.sum(num_mems)), ), dtype=int)
    if num_classes > 2:
        d = np.zeros(shape=(int(np.sum(num_mems)), num_classes), dtype=int)
   
    sum_mems_0 = 0
    sum_mems_1 = 0
    ids_to_indices = {}  # Keeping track of which class ids map to which index
    ids_to_targets = {}  # Keeping track of which class ids map to which target
    for i in range(0, num_distributions):
        # Get the current distributions class id
        class_id = class_ids[i]

        # Record new id: index pair
        if class_id not in ids_to_indices.keys():
            max_index = len(ids_to_indices.values())
            ids_to_indices[class_id] = max_index

        # Set index according to class_id
        class_index = ids_to_indices[class_id]

        # Setting positions and targets of distributions
        sum_mems_1 += num_mems[i]
        x[sum_mems_0:sum_mems_1, :] = rng.normal(centers[i], scales[i],
                                                 size=(num_mems[i], num_dims))
        if num_classes > 2:
            d[sum_mems_0:sum_mems_1, class_index] = 1  # One-hot encoding
        else:
            d[sum_mems_0:sum_mems_1] = class_index  # Binary

        # Record new id: target pair
        if class_id not in ids_to_targets.keys():
            ids_to_targets[class_id] = d[sum_mems_0]

        sum_mems_0 = sum_mems_1
    
    return x, d, ids_to_targets


def standard(x):
    """
    Calculates the means and the standard-deviations of the provided
    data-set independently over each axis.
    
    Parameters
    ----------
    x : nested numpy array of floats
        The data-set, a numpy array of arbitrary-dimensional co-ordinates.

    Returns
    -------
    numpy array of floats
        The means of the data-set, calculated independently over each axis.
    numpy array of floats
        The standard-deviations of the data-set, calculated independently
        over each axis.
    """
    return np.mean(x, axis=0), np.std(x, axis=0)


def standardise(x_trn, x_val=None):
    """
    Standardises x_trn to unit-variance and zero-mean and apply the same
    transformation to x_val (if provided).
    
    Parameters
    ----------
    x_trn : nested numpy array of floats
        The training data-set of arbitrary-dimensional co-ordinates.
    x_val : nested numpy array of floats
        The validation data-set of arbitrary-dimensional co-ordinates.

    Returns
    -------
    nested numpy array of floats
        The standardised training data-set.
    nested numpy array of floats
        The transformed validation data-set (note: this will not
        necessarily have unit-variance and zero-mean).
    """
    mean_trn, std_trn = standard(x_trn)
    x_trn = (x_trn - mean_trn) / std_trn 
    if (isinstance(x_val, type(None))):
        return(x_trn, None)
    x_val = (x_val - mean_trn) / std_trn
    return x_trn, x_val


def generate_datasets(preset_name, presets_file='data_presets.txt',
                      val_mul = 1, try_plot = False,
                      rng=np.random.default_rng()):
    """
    Generates classification data from normal distributions defined according
    to the specified preset.

    Parameters
    ----------
    preset_name : str
        The name of the preset to be loaded from, as given in the specified
        presets text file.
    presets_file : str
        Path to the text file to loads the presets from. Default is
        data_presets.txt.
    val_mul : int
        The relative size (in members) of the validation data-set, with respect
        to the (training data-set). Set val_mul = 0 for no validation data-set.
        Default is 1.
    try_plot : bool
        Whether or not to plot the datasets once generated. Default is False.
    rng : numpy generator
        The random number generator. Default is random unless a rng with a fixed
        seed is provided.

    Returns
    -------
    two tuples of... 
        nested numpy array of floats
            The generated numpy array of I-dimensional co-ordinates.
        numpy array of ints or nested numpy array of ints
            The classifications for each data-point. For binary classifications,
            each target is an integer 0 or 1. For multi-class classification,
            the target uses one-hot encoding.
    """
    #Load the chosen_preset from the presets_file presets from file
    presets = load_presets(presets_file)
    chosen_preset = presets[preset_name]

    # Synthesise data
    x_trn, d_trn, ids_to_targets = generate_class_data(*chosen_preset, rng=rng)
    x_val, d_val, __ = generate_class_data(*chosen_preset, val_mul, rng=rng)

    # Reverse the ids_to_targets dictionary, for plotting labels
    # (Using .tobytes() as dictionaries cannot have arrays for keys)
    targets_to_ids = {targets.tobytes(): ids for ids, targets in
                      ids_to_targets.items()}

    # Plot data
    if try_plot:
        plot_data(x_trn, d_trn, 'training', targets_to_ids)
        plot_data(x_val, d_val, 'validation', targets_to_ids)

    # Standardise data
    x_trn, x_val = standardise(x_trn, x_val)   
    
    return (x_trn, d_trn), (x_val, d_val)


def plot_data(x, d, data_name='generic', targets_to_ids=None):
    """
    Plot 2D input data x, distinguished into the corresponding unique targets d.

    Parameters
    ----------
    x : nested numpy array of floats
        A numpy array of I-dimensional co-ordinates.
    d : numpy array of ints or nested numpy array of ints
        The classifications for each data-point. For binary classifications,
        each target is an integer 0 or 1. For multi-class classification,
        the target uses one-hot encoding.   
    data_name : str
        The name of the data to be used in the plot title.
    targets_to_ids : dictionary of numpy arrays as bytes for keys and strings
                     for values
        Pairs of targets and target labels. The default is None.
        
    Returns
    -------
    None.
    """
    if (isinstance(x, type(None))) or (isinstance(d, type(None))):
        return None

    # Convert numpy arrays to lists (for zipping)
    # Zip the lists, so each co-ordinate is paired to the corresponding target
    # Convert tuples to lists (for sorting)
    dx = zip(d.tolist(), x.tolist())
    dx = list(list(pair) for pair in dx)
    
    # Sort the pairs by the target value
    # Ideally want reverse=False for binary classification and reverse=True 
    # for multi-classification (just aesthetic)
    dx = sorted(dx, reverse=False, key=lambda item: (item[0]))

    # Get the indices that mark the transitions between each unique class
    indices = [index for index, element in enumerate(dx) if element[0] != dx[index-1][0]]
    if 0 not in indices:
        indices.insert(0, 0) # (Only necessary for binary classification)
    indices.append(len(dx))     

    # Convert (back) to numpy arrays, for easy indexing when plotting
    dx = np.asarray(dx, dtype=object)
    for i in range(0, len(dx)):
        dx[i] = np.asarray(dx[i])
        for j in range(0, 2):
            dx[i][j] = np.asarray(dx[i][j])
    
    # Plot each class separately, for automatic labelling and colouring
    fig, ax = plt.subplots()
    ax.set_title('Synthetic ' + data_name + ' data')
    num_classes = len(indices)-1
    for i in range(0, num_classes):
        x = [xy[0] for xy in dx[indices[i]:indices[i+1], 1]]
        y = [xy[1] for xy in dx[indices[i]:indices[i+1], 1]]

        # Constructing the label
        target = dx[indices[i], 0]
        label = str(target)  # Default label if no targets_to_ids provided
        if not isinstance(targets_to_ids, type(None)):
            # Get the class_id from the byte form of the target
            class_id = targets_to_ids[target.tobytes()]

            class_id = str(class_id)
            target = str(target)
            if class_id != target:
                if num_classes > 2:
                    label = class_id + " " + target
                else:
                    label = class_id + " (" + target + ")"

        # Plotting!
        ax.scatter(x, y, label=label)
    
    # Make x and y axis scale equally
    plt.gca().set_aspect('equal', adjustable='box')
                                                   
    ax.legend()

    # Uncomment for origin lines
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)


    # Uncomment for grid lines
    """
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    """

    plt.show()


generate_datasets('uni-brow', try_plot=True)
