import numpy as np
import matplotlib.pyplot as plt

#Lambda (regularization strength) list
lam_list = []
#Performance list
per_list = []

#Convert lists into arrays
lam_array=np.array(lam_list)
per_array=np.array(per_list)

#Make a logarithmic 
lam_array=np.log(lam_array)

#Find the index of the best performance measure (smallest error here)
i=np.asarray(np.where(per_array == np.amin(per_array)))[0,0]

#Print the smallest error and its corresponding 
print('Smallest error', per_array[i], 'given by λ =', lam_array[i])

#Make a scatter plot of the arrays
plt.figure()
plt.scatter(lam_array, per_array)
plt.title('Scatter plot of different lambdas and their performance')
plt.xlabel('log(λ)')
plt.ylabel('Performance')

#Pseudocode:
#def lambda_optimizer(lam_list):
    #for lambd in lam_list:
        #train_model()
        #per_list.append(model.val_loss[-1])