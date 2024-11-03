import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
#--
Iterations = (10**3)
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

#def potiential(x):
#    return 0.5*(x**2)

def wavefunc(x):
    return (np.exp(0.5*-x**2)/np.pi**0.25)**2


def action(path):
    E = 0
    for i in range(1,N):
        E += (0.5*m*(((path[i]-path[i-1])/dt)**2) + 0.5*((path[i]-path[i-1])**2))
    return E*dt

def actionElement(x1,x):
    E = dt*(0.5*m*(((x1-x)/dt)**2) + 0.5*(((x1-x)/2)**2))
    return E


#def metropolis(num,x): #iterates num times to generate optimized paths that start and end at a given x
    numberOfPaths = int(num/100)
    array = np.zeros(shape=(numberOfPaths,N))
    #totalPath = np.zeros(N)
    #create path starting at x
    for j in range(0,numberOfPaths):

        initialPath = np.random.uniform(-3,3,N) # generate random path
        initialPath[0] = initialPath[-1] = x # set starting points to x

        for k in range(0,100):#repeats the thermalization sweep on a path 100 times
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

    return array

def generateNRandomPaths(numberOfPaths,x):
    array = np.zeros(shape=(numberOfPaths,N))
    for j in range(0,numberOfPaths):
        initialPath = np.random.uniform(-3,3,N)
        initialPath[0] = initialPath[-1] = x
        array[j] = initialPath
    return array


def G(paths):# takes in many paths with the same starting point and converts them to one point
    #paths is a 2d array containing all paths
    
    num = len(paths)

    pathSum = 0 #np.zeros(N)

    for i in range(num):
        pathSum += action(paths[i])

    average = pathSum
    #B = np.average(pathSum)
    return average*A

def generateDenominator():
    sum = 0
    for i in range(len(xs2)):
            sum += G(generateNRandomPaths(Iterations,xs2[i]))
    denominator = sum
    return denominator*dx

#print(generateDenominator())
denominator = generateDenominator()


def psi(x):


    numerator = G(generateNRandomPaths(Iterations,x)) #propagator for paths at x, for example: G(metropolis(num,x))
    
    #/num #sum of propagator*dx of path at -3...+3 for example: dx*(G(metropolis(num),-3) + ... + G(metropolis(num),3))

    probability = numerator/denominator

    return probability
a = 0
#print(psi(a),wavefunc(a))


#print(G(generateNRandomPaths(Iterations,1))/generateDenominator())


#print(metropolis(10**4,0))
#print(G(metropolis(Iterations,xs)))

array1 = np.zeros(len(xs))
for i in range(len(xs)):
   array1[i] = psi(xs[i])

plt.ylabel("Probability")
plt.xlabel("X-Position")
plt.plot(xs,array1)
plt.plot(xs,wavefunc(xs))
plt.show()