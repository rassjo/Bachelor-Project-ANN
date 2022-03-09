import numpy as np 

# x is the list of best lambdas for one set of hyperparameters
def SEMcalc(x):
    #Calculate xbar
    xbar=sum(x)/len(x)
    
    #Calculate sample variance
    samvar2=1/(len(x)-1)*(sum(x)**2-2*xbar*sum(x)+xbar**2)
    
    #calculate the standard error of the mean (SEM)
    SEM=np.sqrt(samvar2/len(x))
    
    return SEM