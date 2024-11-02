import numpy as np
import matplotlib.pyplot as plt
import random
###afafafafa
Iterations = (10**4)
N = 7
T = 4
epsilon = T/(N)
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

def energy(x):
    E = 0
    for i in range(1,N):
        E += (0.5*m*(((x[i]-x[i-1])/dt)**2) + potiential((x[i]-x[i-1])/2))
    return E

def generateNRandomPaths(N,x):
    return


def G(x):

    A = (m/(2*np.pi*dt))**(0.5*N)

    for i in range(len(xs)):


    return A

def psi(x):
    denominator = 0
    for i in range(len(x)):
        denominator += G(x[i])*dx
    return G(x)/denominator

def iterate(num,x):

    totalPath = np.zeros(N)
    for k in range(0,num):
        #create path starting at x
        initialPath = np.random.uniform(-3,3,N)
        initialPath[0] = initialPath[-1] = x
        perturbedPath = initialPath
        #perturbs our path at each element to create an optimised path
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
            
            totalPath += perturbedPath
        
    return totalPath/num
        
    

t = np.zeros(N)
for i in range(0,N):
    t[i] = epsilon*i

#print(t)

plt.plot(xs,iterate(Iterations,xs))
plt.plot(xs,wavefunc(xs))
plt.show()