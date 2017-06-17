import numpy as np

x = np.zeros(shape=(2,2))
x[0] = [2, 3]
x[1] = [4, 5]
y = np.zeros(shape=(1, 2))
y[0] = [1, 2]
print( x.T.dot(y) ) 