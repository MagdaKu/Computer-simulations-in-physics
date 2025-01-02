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
def MCS(t, L = 5, N1 = 1000, N2 = 5000, J = 1, susceptibility = False):
    beta = 1/t
    lattice  = np.ones((L,L))
    for i in range(N1): #thermalization
        for j in range(L*L):
            position = (np.random.randint(0, L), np.random.randint(0, L))
            spin_sum = 2 * J * (lattice[boundary_condition(position[0] + 1, position[1], L)] + lattice[
                boundary_condition(position[0] - 1, position[1], L)] + lattice[boundary_condition(position[0], 
                position[1] + 1, L)]  + lattice[boundary_condition(position[0], position[1] - 1, L)])
            r = np.random.uniform(0,1)
            if r < 1/(1 + np.exp(-beta*spin_sum)):
                lattice[position[0], position[1]] = 1
            else:
                lattice[position[0], position[1]] = -1

    magnetization_list = np.empty(N2)
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
    
            magnetization_list[i] = (1 / (L * L)) * np.sum(lattice)

    if susceptibility == True:
        return np.var(magnetization_list) * beta * L**2 
    else:
        return round(np.mean(np.abs(magnetization_list)), 3)

#Task 1

T = [1,2,3,4,5]
results = {}
for t in T:   
    results[t] = MCS(t, L = 5, N1 = 1000, N2 = 5000, J = 1)
print(results)




#Task 2
T_array = list(np.arange(1, 5, 0.1))
result_10 = np.array([])
result_20 = np.array([])
susc_10 = np.array([])
susc_20 = np.array([])
result_analytical = np.array([])
Tc = 2/ np.log(1 + 2**0.5)
J = 1

for t in tqdm(T_array): 
    result_10 = np.append(result_10, MCS(t, L = 10, N1 = 2000, N2 = 5000, J = 1))
    result_20 = np.append(result_20, MCS(t, L = 20, N1 = 2000, N2 = 5000, J = 1))
    susc_10 = np.append(susc_10, MCS(t, L = 10, N1 = 2000, N2 = 5000, J = 1, susceptibility= True))
    susc_20 = np.append(susc_20, MCS(t, L = 20, N1 = 2000, N2 = 5000, J = 1, susceptibility= True))
    if t < Tc:
        result_analytical = np.append(result_analytical, (1 - 1/(np.sinh(2*J/t)**4))**(1/8))

analytical_x = np.linspace(1, Tc, len(result_analytical))  
plt.plot(T_array, result_10, ".", label = "L = 10")
plt.plot(T_array, result_20, ".", label = "L = 20")
plt.plot(analytical_x, result_analytical, label = "Onsager's solution")
plt.title("Magnetisation as a function of temperature ")
plt.xlabel("T")
plt.legend()
plt.show()


plt.plot(T_array, susc_10, ".", label = "L = 10")
plt.plot(T_array, susc_20, ".", label = "L = 20")
plt.title("Susceptibility")
plt.xlabel("T")
plt.legend()
plt.show()






