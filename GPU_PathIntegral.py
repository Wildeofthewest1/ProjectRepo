import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, float64, int64
import math

# Constants and parameters
Iterations = 10**5
N = 7
T = N
m = 1
hbar = 1
xf = 3
xi = -3
dx = 0.05
dt = T / N
_dt = N / T
xs = np.arange(xi, xf, dx)

# Functions to be executed on GPU
@cuda.jit(device=True)
def wavefunc(x):
    return (np.exp(0.5 * -x ** 2) / np.pi ** 0.25) ** 2

@cuda.jit(device=True)
def potential(x):
    return 0.5 * (x ** 2)

@cuda.jit(device=True)
def kinetic(v):
    return 0.5 * m * (v ** 2)

@cuda.jit(device=True)
def S_j(path, i):
    return dt * (kinetic((path[i] - path[i - 1]) * _dt) + potential(0.5 * (path[i] + path[i - 1])))

@cuda.jit(device=True)
def S(path):
    E = 0
    for i in range(1, N):
        E += S_j(path, i)
    return E

# Kernel to perform metropolis sampling on GPU
@cuda.jit
def metropolis_gpu(numberOfPaths, x, paths_out):
    i = cuda.grid(1)
    if i < numberOfPaths:
        # Use a local array instead of creating a device array
        initialPath = cuda.local_array(N, dtype=float64)  # Create a local array on GPU
        # Set the initial path
        initialPath[0] = x
        initialPath[N-1] = x
        for j in range(1, N-1):
            initialPath[j] = 5  # Set default middle points

        perturbedPath = initialPath.copy()

        thermalInterval = 10
        k = 0
        j = 0
        while j < numberOfPaths:
            perturbedPath = initialPath.copy()

            for i in range(1, N - 1):
                old = perturbedPath[i]
                a = S_j(perturbedPath, i) + S_j(perturbedPath, i + 1)
                perturbedPath[i] += 0.5  # Use a fixed perturbation here instead of np.random.uniform
                b = S_j(perturbedPath, i) + S_j(perturbedPath, i + 1)
                actiondiff = b - a

                if actiondiff < 0 or np.random.uniform(0, 1) < np.exp(-actiondiff):  # Use a simple uniform function
                    initialPath[i] = perturbedPath[i]
                else:
                    perturbedPath[i] = old

            k += 1
            if k == thermalInterval:
                paths_out[j] = perturbedPath
                j += 1
                k = 0

# Kernel to sum actions over all paths
@cuda.jit
def G_gpu(paths, result):
    i = cuda.grid(1)
    if i < paths.shape[0]:
        result[i] = np.exp(-S(paths[i]))

# Parallelize Psi on GPU
@cuda.jit
def Psi_gpu(paths, result):
    i = cuda.grid(1)
    if i < paths.shape[0]:
        result[i] = G_gpu(paths[i])

# Main function to run the GPU tasks
def run_on_gpu(numberOfPaths, x_values):
    # Allocate memory on the device for paths and results
    paths_device = cuda.to_device(np.zeros((numberOfPaths, N), dtype=np.float64))
    results_device = cuda.device_array_like(paths_device)

    # Generate paths and copy to GPU
    metropolis_gpu[(numberOfPaths + 255) // 256, 256](numberOfPaths, x_values, paths_device)

    # Calculate G values on GPU
    G_gpu[paths_device.shape[0] // 256 + 1, 256](paths_device, results_device)

    # Copy results back to host
    results_np = results_device.copy_to_host()

    return results_np

# Test the GPU execution
x_values_device = cuda.to_device(xs)
results_np = run_on_gpu(10, x_values_device)

print("Results on GPU:", results_np)


def average(num_to_avs,paths1):
    l = len(xs)
    array = np.zeros(shape=(num_to_avs,l))
    ave = np.zeros(l)
    errs = np.zeros(l)
    analytic = wavefunc(xs)
    #print(array)
    for i in range(num_to_avs):
        array[i] = Psi(paths1)
        ave += array[i]
        print(i)
    for k in range(l):
        errs[k] = np.std(array[:,k])
    averageys = ave/num_to_avs
    aveerrs = errs/np.sqrt(num_to_avs)

    residuals = (averageys-analytic)/np.sqrt(analytic)
    reserrs = aveerrs/np.sqrt(analytic)
    return averageys, aveerrs, residuals, reserrs

#moi = generateNRandomPaths(1,0)[0]
#time = np.arange(0,T,dt)/dt
#plt.plot(moi,time)

f1 = plt.figure(1)

plt.ylabel("Probability")
plt.xlabel("X-Position")
#plt.plot(xs,Psi(paths))

ys, yerrs, normres, reserrs = average(2,5)
plt.errorbar(xs, ys, yerrs, capsize=2)
plt.plot(xs,wavefunc(xs))

f2 = plt.figure(2)

plt.ylabel("Norm-Residuals")
plt.xlabel("X-Position")
stdresarray = np.ones(len(normres))*np.std(normres)
plt.errorbar( xs, normres, reserrs, fmt=".", color = "#8C000F", capsize=2)

for i in range(-3,4):
    plt.plot(xs,i*stdresarray, color='grey', linestyle='dashed')

#actionarray = metropolis(100,0)
#plt.plot(np.arange(0,100,1),actionarray)
plt.show()