import numpy as np

#Lambda (regularization strength) list
llist=[]
#Performance list
plist=[]
#This value is used when finding the smallest lambda from a list
a=10000

#TODO: implement code that appends the regularization strength and performance measure after training here.

#Convert lists into arrays
larray=np.array(llist)
parray=np.array(plist)

#Loop through the performance array
for i in range(len(parray)):
    #Check if the current element is smaller than any previous ones in the array
    if parray[i] < a:
        #If so, save that element and its position in the array.
        a=parray[i]
        b=i

print(larray)
print(parray)
print('Smallest error', parray[b], 'given by', larray[b])