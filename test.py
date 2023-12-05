import numpy as np

def f(x):
    print(x)
    x[0][1, 1] = 0
    print(x)



print(1 ^ 1)
print(1 ^ 0)
print(1 ^ 3)

x = np.array([[0, 1, 2], [0, 4, 5], [0, 0, 8]])
print( np.triu_indices_from(x,k=1) )