"""Synthetic Data Generator

This script contains various procedures for data generation.
"""

import numpy as np
import matplotlib.pyplot as plt


def classification(num_dims, num_mems, centers, scales):
    """Generates data for a specified-dimensional multi-class classification problem from parameter-specified normal distributions.

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

    Returns
    -------
    nested numpy array of floats
        The generated numpy array of I-dimensional co-ordinates.
    numpy array of ints
        The one-hot encoded classifications for each data-point.
    """
    num_classes = len(num_mems)
      
    x = np.empty(shape=(int(np.sum(num_mems)), num_dims), dtype=np.float32)
    d = np.zeros(shape=(int(np.sum(num_mems)), num_classes), dtype=int) #one-hot encoding
        
    sum_mems_0 = 0
    sum_mems_1 = 0
    for i in range(0, num_classes):
        sum_mems_1 += num_mems[i]       
        x[sum_mems_0:sum_mems_1, :] = rng.normal(centers[i], scales[i], size=(num_mems[i], num_dims))
        d[sum_mems_0:sum_mems_1, i] = 1          
        sum_mems_0 = sum_mems_1
    
    return x, d



def standard(x):
    """Calculates the means and the standard-deviations of the provided data-set idendently over each axis.
    
    Parameters
    ----------
    x : nested numpy array of floats
        The data-set, a numpy array of I-dimensional co-ordinates.

    Returns
    -------
    numpy array of floats
        The means of the data-set, calculated independently over each axis.
    numpy array of floats
        The standard-deviations of the data-set, calculated independently over each axis.
    """
    return np.mean(x, axis=0), np.std(x, axis=0)



# Set seed
seed = 5 # Seed should be an integer >= -1
rng = np.random.default_rng(seed) if seed != -1 else np.random.default_rng() # Seed == -1 for random rng, seed >= 0 for fixed rng  
     

# Declare data properties
# Basic circle example
num_dims = 2
num_mems =  np.array([500])
centers = np.array([[0, 0]])
scales = np.array([[5, 5]])
'''
# Intercepting circles example
num_dims = 2
num_mems =  np.array([400, 400])
centers = np.array([[0, 0], [5, 0]])
scales = np.array([[2, 2], [2, 2]])

# Circle in a circle example
num_dims = 2
num_mems =  np.array([300, 100])
centers = np.array([[0, 0], [0, 1]])
scales = np.array([[3, 3], [1, 1]])

# Three ellipses example
num_dims = 2
num_mems =  np.array([300, 300, 300])
centers = np.array([[0, 0], [5, 0], [2.5, 2.5]])
scales = np.array([[3, 1], [1, 3], [3, 1]])

# Five dimensional headache example
num_dims = 5
num_mems =  np.array([100, 100])
centers = np.array([[0, 0, 0, 0, 0], [3, 0, 0, 0, 0])
scales = np.array([[2, 2, 2, 2, 2], [1, 1, 1, 1, 1]])
'''


# Synthesise data
x_trn, d_trn = classification(num_dims, num_mems, centers, scales)
x_val, d_val = classification(num_dims, num_mems * 10, centers, scales)


# Plot training data
if (num_dims == 2): # Only plot 2D data
    plt.figure(0)
    plt.title('Synthetic training data')
      
    num_classes = len(num_mems)
    sum_mems_0 = 0
    sum_mems_1 = 0
    for i in range(0, num_classes): # Plot different categories seperately, such that they are coloured seperately
        sum_mems_1 += num_mems[i]     
        plt.scatter(x_trn[sum_mems_0:sum_mems_1, 0], x_trn[sum_mems_0:sum_mems_1, 1], label = str(d_trn[sum_mems_0]))
        plt.gca().set_aspect('equal', adjustable='box') # Ensure x and y axis scale equally   
        sum_mems_0 = sum_mems_1
        
    plt.legend()
          
        
# Standardise synthesised data (to ready for input into ANN)
mean_trn, std_trn = standard(x_trn)
x_trn = (x_trn - mean_trn) / std_trn # Adjusting inputs to have unit-variance (std = 1) and zero mean (mu = 0)
x_val = (x_val - mean_trn) / std_trn