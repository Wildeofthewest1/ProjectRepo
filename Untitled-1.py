import numpy as np
import matplotlib.pyplot as plt

length = 7
N = 7
T = 4
epsilon = T/N
m = 1
hbar = 1

class randomPath:
    np.random.seed(291823)
    xs = np.random.randint(1,7,8)
    xs[0],xs[-1] = 0,0

def potiential(x):
    return 0.5*x**2
    
print(randomPath.xs)

#--

def energy(x):
    E = 0
    for i in range(1,N):
        E += (0.5*m*((x[i]-x[i-1])/epsilon)**2 + potiential((x[i]-x[i-1])/2))
    return E

def iterate(num):

    initialPath = np.zeros(8)

    for i in range(0,8):
        initialPath[i] = randomPath.xs[i]

    perturbedPath = initialPath

    for j in range(0,num):

        for i in range(0,8):

            init = perturbedPath[i]

            perturbedPath[i] += 0.001 * np.random.randint(1,5,1)[0]

            #print(perturbedPath[i])

            action = (energy(initialPath) - energy(perturbedPath))

            if action < 0: 
                initialPath = perturbedPath
            elif 0.1 * np.random.randint(0,10,1)[0] < np.exp(-epsilon*action):
                initialPath = perturbedPath
            else:
                perturbedPath[i] = init
            
    return perturbedPath

t = np.zeros(8)
for i in range(0,8):
    t[i] = randomPath.xs[i]

#print(iterate(1000))


plt.plot(t,t)
plt.show()