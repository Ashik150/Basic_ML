import numpy as np

f = np.array([1,-2,0,4])

def f(x):
    return np.polyval(f, x) 

def bisection_method(f, xl, xu, epsilon, iterations):
    for i in range(iterations):
        x_root = (xl + xu) / 2
        if f(xl) * f(x_root) < 0:
            xu = x_root
        else:
            xl = x_root
        if abs(f(x_root)) < epsilon:
            break
      
    return x_root

#Write a driver code here for calling the function and testing it
xl = -2
xu = 3
epsilon = 0.005
MAX_ITERS = 100
print(bisection_method(f, xl, xu, epsilon, MAX_ITERS))