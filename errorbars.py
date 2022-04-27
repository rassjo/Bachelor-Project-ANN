import numpy as np 
import sys
import matplotlib.pyplot as plt

# Input whether or not the plot should be linear or log10.
plot_scale = None
while not (plot_scale == "linear" or plot_scale == "log10"):
    plot_scale = input("Do you want the plot to be 'linear' or 'log10'? ")
log_plot = plot_scale == "log10"

# Define function that calculates the error bars
def SEMcalc(x):
    #Note: x should be an array of the best lambdas for one set of hyperparameters
    N = len(x)
    #Calculate xbar
    mean=meancalc(x)
    
    #Calculate sample variance
    samvar2=1/(N-1)*sum((x-mean)**2)
    #print(samvar2)
    #calculate the standard error of the mean (SEM)
    SEM=np.sqrt(samvar2/N)
    
    return SEM

# Define function that calculates the mean of a list
def meancalc(x):
    return sum(x)/len(x)

all_lambda = {}

# Find the data files
identifier = '2ce8e323d822'
seeds = [10,20,30,40,50,60,70,80,90,100]

# Extract lambda values from data files and append the into lists
for seed in seeds:
    with open(f'patterns_lambda_{identifier}_{seed}.txt', 'r') as data:
        for line in data:
            if line[0] != '$':
                continue
            values = line.split(';')
            lambd = float(values[0][2:-1])
            hp_value = float(values[1][1:-1])
            # Append logarithmic lambda value if log10 is chosen
            if (log_plot):
                try:
                    all_lambda[hp_value].append(np.log10(lambd))
                except:
                    all_lambda[hp_value] = []
                    all_lambda[hp_value].append(np.log10(lambd))
            # Otherwise append the linear lambda values
            else:                    
                try:
                    all_lambda[hp_value].append(lambd)
                except:
                        all_lambda[hp_value] = []
                        all_lambda[hp_value].append(lambd)


# Construct the x and y values, as well as calculate error bars
x_axis = list(all_lambda.keys())
y_axis = [meancalc(np.array(all_lambda[point])) for point in x_axis]
error_bars = [SEMcalc(np.array(all_lambda[point])) for point in x_axis]  


if (log_plot):
    # Convert x values to logarithmic form (y-axis are already logarithmic if log10 is chosen)
    x_axis = np.log10(x_axis)
    
    # Create linear regression
    coef = np.polyfit(x_axis, y_axis, 1) # Create a least squares polynomial fit from the log-axes.
    linear_regression = np.poly1d(coef) # Make linear_regressoin a function which takes in x (log) and returns an estimate for y (log)
    # Plot linear regression, and print the gradient and y-intercept.
    plt.plot(x_axis, linear_regression(x_axis), '--k', label="Linear Regression (Least Squares)")
    print("gradient = " + str(linear_regression.coefficients[0]))
    print("$y$-intercept = " + str(linear_regression.coefficients[1]))
    
    
#Plot the values and their error bars
plt.errorbar(x_axis, y_axis, yerr=error_bars, fmt='o', color='r', ecolor='r', capsize=5, label='best lambdas vs # patterns')
plt.xlabel(plot_scale + ' # patterns')
plt.ylabel(plot_scale + ' best lambdas')
plt.title('best lambdas vs # patterns ')
plt.legend()
plt.savefig('L2_Sigma_1_plot_'+ str(plot_scale) + '.png')
plt.show()
plt.clf()

""" 
# An attempt to do fancy custom limits, so that the start and end of the linear regression is not seen.
# However, having trouble getting it look quite right without putting in more effort,
# as it is not taking into account the error bars when determining the top (and bottom).
left = min(x_axis)
right = max(x_axis)
delta_x = right - left
bottom = min(y_axis)
top = max(y_axis)
delta_y = top-bottom
print(delta_y)
plt.xlim(left - delta_x*0.15, right + delta_x*0.15)
plt.ylim(bottom - delta_y*0.15, top + delta_y*0.15)
"""
    