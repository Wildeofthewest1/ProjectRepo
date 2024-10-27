import numpy as np
import matplotlib.pyplot as plt
import random


N = 7
T = 4
epsilon = T/N
m = 1
hbar = 1

class randomPath:
    np.random.seed(456486823)
    xs = np.random.randint(1,8,N+1)
    xs[0],xs[-1] = 0,0

def potiential(x):
    return 0.5*(x**2)

def wavefunc(x):
    return (np.exp(0.5*-x**2)/np.pi**0.25)**2

#--

def energy(x):

    E = 0

    for i in range(1,N+1):
        E += (0.5*m*(((x[i]-x[i-1])/epsilon)**2) + potiential((x[i]-x[i-1])/2))
    return E

def iterate(num):

    initialPath = np.zeros(N+1)

    for i in range(0,N+1):
        initialPath[i] = randomPath.xs[i]

    perturbedPath = initialPath

    for j in range(0,num):

        for i in range(1,len(initialPath)-1):

            rand = random.random()
            init = perturbedPath[i]
            a = energy(perturbedPath)
            perturbedPath[i] += 1*rand
            b = energy(perturbedPath)

            action = epsilon * (b-a)

            if action < 0: 
                initialPath = perturbedPath
            elif 0.1*random.randint(0,10) < np.exp(-action):
                initialPath = perturbedPath
            else:
                perturbedPath[i] = init
            
    return perturbedPath/num

t = np.zeros(N+1)
for i in range(0,N+1):
    t[i] = epsilon*i

plt.plot(t,iterate(10**5))
#plt.plot(t,wavefunc(t))
plt.show()