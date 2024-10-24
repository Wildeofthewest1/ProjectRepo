#s;dfljsa;dlfj
import numpy as np
import matplotlib.pyplot as plt

length = 30
N = 7
T = 4
epsilon = T/N

def potiential(x):
    return 0.5*x**2

class randomPath:
    xs = np.random.randint(1,N,length)
    xs[0],xs[-1] = 0,0
    
print(randomPath.xs)