import numpy as np
import matplotlib.pyplot as plt

#Lambda (regularization strength) list
llist=[]
#Performance list
plist=[]

#Example lists, remove when implementing this code into your program
llist=[1, 10, 100, 1000, 10000]
plist=[0.8, 0.6, 0.5, 0.7, 0.9]

#TODO: implement code that appends the regularization strength and performance measure after training here.

#Convert lists into arrays
larray=np.array(llist)
parray=np.array(plist)
#Make a logarithmic 
loglarray=np.array(np.log(larray))


#Find the index of the best performance measure (smallest error here)
i=np.asarray(np.where(parray == np.amin(parray)))[0,0]

#Print the smallest error and its corresponding 
print('Smallest error', parray[i], 'given by λ =', larray[i])


#Make a scatter plot of the arrays
plt.figure()
plt.scatter(loglarray, parray)
plt.title('Scatter plot of different lambdas and their performance')
plt.xlabel('log(λ)')
plt.ylabel('Performance')
plt.show()
