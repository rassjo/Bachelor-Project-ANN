import ANN as ann
import numpy as np
import activation_functions as act
import synthetic_data_generation as sdg
import training_statistics as ts
import matplotlib.pyplot as plt

lambdx = []
#notrainy = []
trainy = []
valy = []
for la in range(0, 10):
    #------------------------------------------------------------------------------
    # Create random number generators:
    # seed == -1 for random rng, seed >= 0 for fixed rng (seed should be integer)
    data_seed = 5
    ann_seed = data_seed
    
    def generate_rng(seed):
        # seed should be an integer
        # -1 is for random rng, integer >= 0 for fixed
        if (seed != -1):
            return np.random.default_rng(data_seed)
        return np.random.default_rng()
    
    data_rng = generate_rng(data_seed)
    ann_rng = generate_rng(ann_seed)
    
    #------------------------------------------------------------------------------
    # Import data
    trn, val = sdg.generate_datasets('circle_ception', try_plot = True,
                                     rng = data_rng)    
    input_dim = len(trn[0][0]) #Get the input dimension from the training data
    #------------------------------------------------------------------------------
    lambd = 0.002 + la*0.002 #(2 inputs, 200 patterns -> conts*1/100 as size for lambda optimally? (for as good validation performance as possible)) 
                    
    print(lambd)
    #Properties of all the layers
    #Recipe for defining a layer: [number of nodes, activation function, L2]
    layer_defines = [[20, act.tanh, lambd],
                     [20, act.tanh, lambd],
                     [20, act.tanh, lambd],
                     [1, act.sig, lambd]]

    test = ann.Model(input_dim, layer_defines, ann_rng)
    
    #answer1 = check_results(test) #Loss without training
    
    test.train(trn,val,0.1,40) #training, validation, lrn_rate, epochs, minibatchsize=0
    
    answer2 = ann.check_results(test) #Loss after training
    
    validation = test.history['val'][-1] #Loss for validation
    
    lambdx.append(lambd)
    #notrainy.append(answer1)
    trainy.append(answer2)
    valy.append(validation)

plt.figure()
#plt.plot(lambdx, notrainy, 'ro', label='error before train vs lambd') #consider using (x,y,linestyle = 'none',marker = '.',markersize= 0.1)
plt.plot(lambdx, trainy, 'go', label='error after train vs lambd')
plt.plot(lambdx, valy, 'bo', label='Validation error vs lambd')
plt.xlabel('lambdas')
plt.ylabel('Error')
plt.title('Error over lambdas')
plt.legend()
plt.savefig('ErrorPlot.png')
plt.show()   












