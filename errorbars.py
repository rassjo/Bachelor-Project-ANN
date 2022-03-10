import numpy as np 
import matplotlib.pyplot as plt

def SEMcalc(x):
    #Note: x should be an array of the best lambdas for one set of hyperparameters
    N = len(x)
    #Calculate xbar
    mean=meancalc(x)
    
    #Calculate sample variance
    samvar2=1/(N-1)*sum((x-mean)**2)
    print(samvar2)
    #calculate the standard error of the mean (SEM)
    SEM=np.sqrt(samvar2/N)
    
    return SEM

def meancalc(x):
    return sum(x)/len(x)

all_lambda = {}

identifier = '8d01fe5a229'
seeds = [2,3,4]

for seed in seeds:
    with open(f'patterns_lambda_{identifier}_{seed}.txt', 'r') as data:
        for line in data:
            if line[0] != '$':
                continue
            values = line.split(';')
            lambd = float(values[0][2:-1])
            hp_value = float(values[1][1:-1])
            try:
                all_lambda[hp_value].append(lambd)
            except:
                all_lambda[hp_value] = []
                all_lambda[hp_value].append(lambd)
                
x_axis = list(all_lambda.keys())
y_axis = [meancalc(np.array(all_lambda[point])) for point in x_axis]
error_bars = [SEMcalc(np.array(all_lambda[point])) for point in x_axis]

plt.figure()
plt.errorbar(x_axis, y_axis, yerr=error_bars, fmt = 'o', color='r', ecolor='b', capsize=5, label='best lambdas vs # patterns')
plt.xlabel('# patterns')
plt.ylabel('best lambdas')
plt.title('best lambdas vs # patterns')
plt.legend()
plt.savefig('patterns_lambda_plot.png')
plt.show()
plt.clf()
    