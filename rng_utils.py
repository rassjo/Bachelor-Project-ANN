import numpy as np

def generate_rng(seed):
    # seed should be an integer
    # -1 is for random rng, integer >= 0 for fixed
    if (seed != -1):
        return np.random.default_rng(seed)
    else:
        print('Warning: Please set a fixed seed, otherwise results will not be reproducable!')
        return np.random.default_rng()