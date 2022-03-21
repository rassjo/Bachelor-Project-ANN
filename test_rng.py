import numpy as np

def generate_rng(seed):
    # seed should be an integer
    # -1 is for random rng, integer >= 0 for fixed
    if (seed != -1):
        return np.random.default_rng(seed)
    else:
        return np.random.default_rng()
        #raise Exception(f'Please set a fixed seed, otherwise results will not be reproducable!')

search_rng = generate_rng(-1)

#print(1-search_rng.random())

#mean = -3
#sigma = 3.5
#l2 = search_rng.lognormal(mean = mean, sigma = sigma)
#while l2 >= 1:
#    l2 = search_rng.lognormal(mean = mean, sigma = sigma)

#mean = 0.01
#scale = 0.5
#print(search_rng.logistic(loc = mean, scale = scale))


#print(search_rng.normal())

# Log Uniform
l2_range = [10e-6, 1]

l2_pow_range = np.log10(l2_range)

l2_pow = search_rng.random() # Generate random number from the half-open interval [0, 1)
l2_pow *= (l2_pow_range[1] - l2_pow_range[0]) # Reduce the range
l2_pow += l2_pow_range[0] # Displace the range

l2 = 10**l2_pow

print(l2)

#a = 0.1
#b = 4
#log_uniform = lambda x : 1 / (x * np.log10(b / a))
#log_uniform = lambda x : (l2_range[1] - l2_range[0])*np.log10(x)