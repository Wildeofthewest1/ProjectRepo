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
xs2 = np.linspace(xi1,xf1,TotalXs1)
A = (m/(2*np.pi*dt))**(0.5*N)
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

def metropolis(num,x): #iterates num times to generate optimized paths that start and end at a given x
    numberOfPaths = num
    n = 100
    array = np.zeros(shape=(numberOfPaths,N))

    for j in range(numberOfPaths):

        initialPath = np.random.uniform(-3,3,N) # generate random path
        initialPath[0] = initialPath[-1] = x # set starting points to x

        for k in range(n):#repeats the thermalization sweep on a path n times
            perturbedPath = initialPath
            
            for i in range(1,len(initialPath)-1): #perturbs our path at each element to create a slightly more optimised path

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
                
        array[j] = perturbedPath
        #print("Path added",j)
    #print("Paths Generated")
    print(array)
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



metro = False

def generateDenominator(): #sum of propagator*dx of path at -3...+3 for example: dx*(G(metropolis(num),-3) + ... + G(metropolis(num),3))
    sum = 0
    for i in range(len(xs2)):
            if not metro:
                sum += G(generateNRandomPaths(Iterations,xs2[i]))
            else:
                sum += G(metropolis(Iterations,xs2[i]))
    denominator = sum
    return denominator*dx

#denominator = generateDenominator()
#print(denominator)
#denominator = 13.424550924328017
denominator = 136.22891862560073
def psi(x):

    if not metro:
        numerator = G(generateNRandomPaths(Iterations,x)) #brute force paths
    else:
        numerator = G(metropolis(Iterations,x)) #metropolis paths
    probability = numerator/denominator
    #print("Probability added")
    return probability

#print(metropolis(10**4,0))
#print(G(metropolis(Iterations,xs)))
def generatePlot():
    array1 = np.zeros(len(xs))
    for i in range(len(xs)):
        array1[i] = psi(xs[i])
    return plt.plot(xs,array1)

plt.ylabel("Probability")
plt.xlabel("X-Position")
generatePlot()
plt.plot(xs,wavefunc(xs))
plt.show()