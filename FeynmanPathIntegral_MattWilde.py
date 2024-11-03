import numpy as np
import matplotlib.pyplot as plt
import random
###afafafafa
Iterations = (10**3)
N = 20
T = 4
#epsilon = T/(N)
m = 1
hbar = 1

xf = 2
xi = 0
xrange = xf-xi
dx = 0.05
dt = T/N
xs = np.linspace(xi,xf,int(((xrange)/dx)+1))
print("xs =", xs)


def potiential(x):
    return 0.5*(x**2)

def wavefunc(x):
    return (np.exp(0.5*-x**2)/np.pi**0.25)**2

#--

def action(path):
    E = 0
    for i in range(1,N):
        E += (0.5*m*(((path[i]-path[i-1])/dt)**2) + potiential((path[i]-path[i-1])/2))
    return E*dt

def generateNRandomPaths(N,x):
    return


def G(paths):
    #paths is a 2d array containing all paths
    A = (m/(2*np.pi*dt))**(0.5*N)
    tot = 0
    num = len(paths)
    for i in range(num):
        tot += dx*np.exp(-action(paths[i]))
    return A*tot

def psi(x):
    denominator = 1

    return G(x)/denominator


def metropolis(num,x):
    set = np.zeros(shape=(len(x),N))
    for array in range(len(x)):
        totalPath = np.zeros(N)
        for k in range(0,num):
            #create path starting at x
            initialPath = np.random.uniform(-3,3,N)
            initialPath[0] = initialPath[-1] = x[array]
            perturbedPath = initialPath
            #perturbs our path at each element to create a slightly more optimised path
            for i in range(1,len(initialPath)-1):
                
                init = perturbedPath[i]
                a = action(perturbedPath)
                perturbedPath[i] += np.random.uniform(0,1)
                b = action(perturbedPath)

                actiondiff = (b-a)

                if actiondiff < 0: 
                    initialPath = perturbedPath
                elif np.random.uniform(0,1) < np.exp(-actiondiff):
                    initialPath = perturbedPath
                else:
                    perturbedPath[i] = init
                
                totalPath += perturbedPath
        set[array] = totalPath/num

    return set
        
#print(metropolis(10**4,0))
#print(G(metropolis(Iterations,xs)))



plt.plot(xs,psi(xs))
plt.plot(xs,wavefunc(xs))
plt.show()