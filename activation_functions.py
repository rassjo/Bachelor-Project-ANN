import numpy as np

#------------------------------------------------------------------------------
#Linear
#------------------------------------------------------------------------------
linear_def = lambda a: a
d_linear_def = lambda y: 1
#------------------------------------------------------------------------------
#Rectified Linear Unit
#------------------------------------------------------------------------------
ReLU_def = lambda a: max(0.0,a)
#NOTE: If we use 0 instead of 0.0 numpy will return an int32 array when the
#first entry in the array is 0, meaning it rounds the other numbers down to
#the closest integer as well!
def d_ReLU_def(y):
    if y > 0:
        return 1.0
    else:
        return 0.0
#------------------------------------------------------------------------------
#Sigmoid (logistic)
#------------------------------------------------------------------------------
sigmoid_def = lambda a: 1/(1+np.e**(-a))
d_sigmoid_def = lambda y: y*(1.0 - y)
#------------------------------------------------------------------------------
#Tangens Hyperbolicus
#------------------------------------------------------------------------------
tanh_def = lambda a: np.tanh(a)
d_tanh_def = lambda y: 1.0 - y**2
#------------------------------------------------------------------------------

#Vectorize allows the function to act elementwise on arrays
lin = {'act':np.vectorize(linear_def),'der':np.vectorize(d_linear_def)}
ReLU = {'act':np.vectorize(ReLU_def),'der':np.vectorize(d_ReLU_def)}
sig = {'act':np.vectorize(sigmoid_def),'der':np.vectorize(d_sigmoid_def)}
tanh = {'act':np.vectorize(tanh_def),'der':np.vectorize(d_tanh_def)}