import numpy as np 



# x is the list of best lambdas for one set of hyperparameters
def SEMcalc(x):
    #Note: x should be an array
    N = len(x)
    #Calculate xbar
    mean=sum(x)/N
    
    #Calculate sample variance
    samvar2=1/(N-1)*sum((x-mean)**2)
    print(samvar2)
    #calculate the standard error of the mean (SEM)
    SEM=np.sqrt(samvar2/N)
    
    return SEM

