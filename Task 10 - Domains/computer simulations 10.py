import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit

@jit(nopython=True)
def boundary_condition(x,y, L):
    if x >= L:
        x = x - L
    elif x < 0:
        x = x + L
    if y >= L:
        y = y - L
    elif y < 0:
        y = y + L
    return (x, y)


@jit(nopython=True)
def MCS(t, L = 5, N2 = 5000, J = 1, susceptibility = False):
    beta = 1/t
    lattice = np.random.choice(np.array([-1, 1]), size=(L, L))
    for i in range(N2): #measurements
        for j in range(L*L):
            spin = np.random.randint(0, L**2)
            position = (spin // L, spin%L)
            spin_sum = 2 * J * (lattice[boundary_condition(position[0] + 1, position[1], L)] + lattice[
                boundary_condition(position[0] - 1, position[1], L)] + lattice[boundary_condition(position[0], 
                position[1] + 1, L)]  + lattice[boundary_condition(position[0], position[1] - 1, L)])
            r = np.random.uniform(0,1)
            if r < 1/(1 + np.exp(-beta*spin_sum)):
                lattice[position[0], position[1]] = 1
            else:
                lattice[position[0], position[1]] = -1           
    return lattice 

#Task 1
T = 2
J = 1
L = 500

N2_array = np.array([10,100,1000,5000])

for N2 in tqdm(N2_array): 
    lattice =  MCS(T, L = L, N2 = N2, J = J)

    plt.imshow(lattice, cmap='gray', interpolation='nearest')
    plt.colorbar() 
    plt.title(f'Lattice Configuration for {N2} Monte Carlo Steps')
    plt.show()

    chi = np.zeros(L//2) # correlation function
    for r in range(L//2):
        chi_sum = 0
        for r0 in range(L):
            chi_sum += lattice[:,r0]*lattice[:, (r0+r) % L]/L
        chi[r] = np.mean(chi_sum) # extra averaging over rows

    r_values = np.arange(len(chi))  # Corresponding distance values
    plt.plot(r_values, chi, marker='o', linestyle='-', color='b')
    plt.xlabel('Distance r')
    plt.ylabel('Spatial Correlation chi(r)')
    plt.title(f'Spatial Correlation Function, MCS = {N2}')
    plt.grid(True)
    plt.show()

    

    
