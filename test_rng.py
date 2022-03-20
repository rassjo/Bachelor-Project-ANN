import numpy as np

def generate_rng(seed):
    # seed should be an integer
    # -1 is for random rng, integer >= 0 for fixed
    if (seed != -1):
        return np.random.default_rng(seed)
    else:
        raise Exception(f'Please set a fixed seed, otherwise results will not be reproducable!')

search_rng = generate_rng(4)

#print(1-search_rng.random())

mean = -3
sigma = 3.5
l2 = search_rng.lognormal(mean = mean, sigma = sigma)
while l2 >= 1:
    l2 = search_rng.lognormal(mean = mean, sigma = sigma)

#mean = 0.01
#scale = 0.5
#print(search_rng.logistic(loc = mean, scale = scale))


#print(search_rng.normal())
