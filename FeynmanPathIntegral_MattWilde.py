import numpy as np
import matplotlib.pyplot as plt
import random
###afafafafa
Iterations = (10**3)
N = 20
T = 4
epsilon = T/(N)
m = 1
hbar = 1

class randomPath:
    #np.random.seed(456486823)
    xs = np.random.uniform(0,7,N)
    xs[0] = xs[-1] = 0
    #print(xs)



def potiential(x):
    return 0.5*(x**2)

def wavefunc(x):
    return (np.exp(0.5*-x**2)/np.pi**0.25)**2

#--

def energy(x):
    E = 0
    for i in range(1,N):
        E += (0.5*m*(((x[i]-x[i-1])/epsilon)**2) + potiential((x[i]-x[i-1])/2))
    return E




def iterate(num):

    initialPath = np.zeros(N)

    for i in range(0,N):
        initialPath[i] = randomPath.xs[i]

    perturbedPath = initialPath

    for j in range(0,num):

        for i in range(1,len(initialPath)-1):

            init = perturbedPath[i]
            a = energy(perturbedPath)
            perturbedPath[i] += np.random.uniform(0,1)
            b = energy(perturbedPath)

            action = epsilon * (b-a)

            if action < 0: 
                initialPath = perturbedPath
            elif np.random.uniform(0,1) < np.exp(-action):
                initialPath = perturbedPath
            else:
                perturbedPath[i] = init
            
    return perturbedPath/N

t = np.zeros(N)
for i in range(0,N):
    t[i] = epsilon*i

#print(t)

plt.plot(t,iterate(Iterations))
#plt.plot(t,wavefunc(t))
plt.show()