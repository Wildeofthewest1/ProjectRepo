import numpy as np
import matplotlib.pyplot as plt

#--
Iterations = (10**5)
N = 7
T = 4
#epsilon = T/(N)
m = 1
hbar = 1

xf = 3
xi = -3
xrange = xf-xi
dx = 0.05
dt = T/N
TotalXs = int(((xrange)/dx)+1)

xf1 = 3
xi1 = -3
xrange1 = xf1-xi1
TotalXs1 = int(((xrange1)/dx)+1)

xs = np.linspace(xi,xf,TotalXs)
#A = (m/(2*np.pi*dt))**(0.5*N)
#print("xs =", xs)

def wavefunc(x):
    return (np.exp(0.5*-x**2)/np.pi**0.25)**2

def potential(x):
    return 0.5*(x**2)

def action(path):
    E = 0
    for i in range(1,N):
        E += (0.5*m*(((path[i]-path[i-1])/dt)**2) + potential(0.5*(path[i]+path[i-1])))
    return E*dt

def metropolis(numberOfPaths,x): #iterates num times to generate optimized paths that start and end at a given x

    thermalInterval = 10

    array = np.zeros(shape=(numberOfPaths,N))
    initialPath = np.random.uniform(-3,3,N) # generate random path
    initialPath[0] = initialPath[-1] = x # set starting points to x

    j = 0
    k = 0
    while j < numberOfPaths:

        perturbedPath = initialPath
        
        for i in range(1,N-1): #perturbs our path at each element to create a slightly more optimised path

            old = perturbedPath[i]
            a = action(perturbedPath)
            perturbedPath[i] += np.random.uniform(0,2)-1
            b = action(perturbedPath)
            actiondiff = (b-a)

            if actiondiff < 0: 
                initialPath = perturbedPath
            elif np.random.uniform(0,1) < np.exp(-actiondiff):
                initialPath = perturbedPath
            else:
                perturbedPath[i] = old
        
        k += 1
        if k == thermalInterval:
            array[j] = perturbedPath
            j+=1
            k = 0

    return array

def generateNRandomPaths(numberOfPaths,x):
    array = np.zeros(shape=(numberOfPaths,N))
    for j in range(0,numberOfPaths):
        initialPath = np.random.uniform(-3,3,N)
        initialPath[0] = initialPath[-1] = x
        array[j] = initialPath
    return array

def G(paths):#sums action of all paths
    #paths is a 2d array containing all paths
    num = len(paths)
    pathSum = 0
    for i in range(num):
        pathSum += np.exp(-action(paths[i]))
        
    return pathSum

metro = True
n = 100

def generatePlot():
    array1 = np.zeros(len(xs))
    sum = 0
    for i in range(len(xs)):
        if not metro: array1[i] = G(generateNRandomPaths(Iterations,xs[i])) #brute force paths
        else: array1[i] = G(metropolis(n,xs[i])) #metropolis paths
        sum += array1[i]
        print("{:0.2f}% complete".format(100*(i+1)/len(xs)))
    return plt.plot(xs,array1/(sum*dx))

#moi = generateNRandomPaths(1,0)[0]
#time = np.arange(0,T,dt)/dt
#plt.plot(moi,time)


plt.ylabel("Probability")
#plt.ylabel("Time")
plt.xlabel("X-Position")
generatePlot()
plt.plot(xs,wavefunc(xs))
plt.show()