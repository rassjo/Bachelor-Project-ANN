"""Synthetic Data Generator

This script contains various procedures for data generation.

TO DO: CONSIDER OOPIFYING EVERYTHING / DATA PROPERTIES AT LEAST.
"""

import numpy as np
import matplotlib.pyplot as plt

def load_datasets(file_name):
    
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
                #Iterate over all the classes that we have defined members for
                members = np.array([int(number) for number in properties[2].split('/')])
                #Split up the coordinates for the different classes
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
                #Add the dataset to the dictionary
                datasets[name] = [dimensions, members, centers, scales]
    
    return datasets


def generate_class_data(num_dims, mems, centers, scales, val=1):
    """Generates data for a specified-dimensional multi-class classification
    problem from parameter-specified normal distributions.

    Parameters
    ----------
    num_dims : int
        The number of dimensions for the data to be generated in.
    num_mems : numpy array of integers
        The number of members of each class.
    centers : nested numpy array of floats
        The num_dims-dimensional displacements of each class.
    scales : nested numpy array of floats
        The num_dims-dimensional standard-deviation (scale) of each class.
    val: int
        The relative size of the validation data set, use only when creating
        a valdiation data set from the same distribution to go with the
        training data

    Returns
    -------
    nested numpy array of floats
        The generated numpy array of I-dimensional co-ordinates.
    numpy array of ints
        The one-hot encoded classifications for each data-point.
    """
    num_mems = mems*val
    #We have to define a new variable explicitly, otherwise if we just
    #modify the argument that we sent to the function we actually modify
    #the original array and the changes remain after we leave the function
    #because arrays are fucking stupid.
    
    num_classes = len(num_mems)
      
    x = np.empty(shape=(int(np.sum(num_mems)), num_dims), dtype=np.float32)
    d = np.zeros(shape=(int(np.sum(num_mems)), ), dtype=int)
    if (num_classes > 2):
        d = np.zeros(shape=(int(np.sum(num_mems)), num_classes), dtype=int)
   
    sum_mems_0 = 0
    sum_mems_1 = 0
    for i in range(0, num_classes):
        sum_mems_1 += num_mems[i]       
        x[sum_mems_0:sum_mems_1, :] = rng.normal(centers[i], scales[i],
                                                 size=(num_mems[i], num_dims))
        if (num_classes > 2):
            d[sum_mems_0:sum_mems_1, i] = 1   
        else:
            d[sum_mems_0:sum_mems_1] = i
        sum_mems_0 = sum_mems_1
    
    return x, d

def standard(x):
    """Calculates the means and the standard-deviations of the provided
    data-set independently over each axis.
    
    Parameters
    ----------
    x : nested numpy array of floats
        The data-set, a numpy array of I-dimensional co-ordinates.

    Returns
    -------
    numpy array of floats
        The means of the data-set, calculated independently over each axis.
    numpy array of floats
        The standard-deviations of the data-set, calculated independently
        over each axis.
    """
    return np.mean(x, axis=0), np.std(x, axis=0)


# Set seed
# Seed == -1 for random rng, seed >= 0 for fixed rng
seed = -1 # Seed should be an integer
rng = np.random.default_rng(seed) if seed != -1 else np.random.default_rng()

#Load datasets from file
datasets = load_datasets('datasets.txt')

# Which dataset to test
chosen_dataset = datasets['smiley']

# Synthesise data
x_trn, d_trn = generate_class_data(*chosen_dataset)
x_val, d_val = generate_class_data(*chosen_dataset, val = 10)

num_dims = chosen_dataset[0]
num_mems = chosen_dataset[1]

# Plot training data
if (num_dims == 2): # Only plot 2D data
    plt.figure(0)
    plt.title('Synthetic training data')

    num_classes = len(num_mems)
    sum_mems_0 = 0
    sum_mems_1 = 0
    for i in range(0, num_classes): # Plot different categories seperately,
                                    # such that they are coloured seperately
        sum_mems_1 += num_mems[i]     
        plt.scatter(x_trn[sum_mems_0:sum_mems_1, 0],
                    x_trn[sum_mems_0:sum_mems_1, 1],
                    label = str(d_trn[sum_mems_0]))
        plt.gca().set_aspect('equal', adjustable='box') # Ensure x and y axis
                                                        # scale equally   
        sum_mems_0 = sum_mems_1
        
    plt.legend()
          
# Standardise synthesised data (to ready for input into ANN)
# Adjusting inputs to have unit-variance and zero mean.
mean_trn, std_trn = standard(x_trn)
x_trn = (x_trn - mean_trn) / std_trn 
x_val = (x_val - mean_trn) / std_trn