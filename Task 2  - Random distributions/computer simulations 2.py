import numpy as np
import matplotlib.pyplot as plt 


#Task 1
def random_walk(N, t):
    x_particles = np.zeros(N)
    particle_example = np.array([])
    for i in range(t):
        delta = np.random.randn(N)
        x_particles += delta
        particle_example = np.append(particle_example, x_particles[0])
    return particle_example, x_particles

N = 10000
t = 1000
t_array = np.linspace(0,1,t)

particle_example_1, x_particles_1 = random_walk(N, t)
particle_example_2, x_particles_2 = random_walk(N, t)
particle_example_3, x_particles_3 = random_walk(N, t)
particle_example_4, x_particles_4 = random_walk(N, t)
particle_example_5, x_particles_5 = random_walk(N, t)
                             
plt.figure()
plt.title("Trajectory of a particle")
plt.xlabel("Time")
plt.ylabel("Trajectory")
plt.grid()
plt.plot(t_array, particle_example_1,  label = "Particle 1")
plt.plot(t_array, particle_example_2,  label = "Particle 2")
plt.plot(t_array, particle_example_3,  label = "Particle 3")
plt.plot(t_array, particle_example_4,  label = "Particle 4")
plt.plot(t_array, particle_example_5,  label = "Particle 5")
plt.legend()
plt.show()


plt.figure()
plt.hist(x_particles_1, density = True, bins = 'auto')
plt.title(f"Histogram of final positions after {t} steps")
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, N)
prediction = 1/ (2*np.pi*t)**0.5 * np.exp(-x**2 / (2*t))
plt.plot(x, prediction, label = "Analytical prediction")
plt.grid()
plt.legend()
plt.show()





#Task 2
N = 5000
C = 20
T = int(1e05)
particles_energy = np.ones(N) * C

for i in range(T):
    first_particle = np.random.randint(N)
    second_particle = np.random.randint(N)
    while first_particle == second_particle: # pick again if the same particle is chosen
        second_particle = np.random.randint(N) 
    whole_energy = particles_energy[first_particle] + particles_energy[second_particle]
    distribution = np.round(np.random.rand(),5) #distribution of energy 
    particles_energy[first_particle] = distribution * whole_energy
    particles_energy[second_particle] = (1 - distribution) * whole_energy


plt.figure()
plt.hist(particles_energy, density = True, bins = 'auto')
plt.title("Particles energy")
plt.grid()
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, N)
prediction = 1/C * np.exp(-x/C)
plt.plot(x, prediction, label = "Boltzmann probability distribution")
plt.legend()
plt.show()



#extra - mean square displacement
N = 1000
t = 1000
t_array = np.arange(t)
x_particles = np.zeros(N)
y_particles = np.zeros(N)
displacement_1 = np.array([])
displacement_2 = np.array([])

for i in range(t):
    delta = np.random.randn(N)
    x_particles += delta
    delta = np.random.randn(N)
    y_particles += delta
    MSD_1 = np.mean(x_particles**2)
    MSD_2 = np.mean((x_particles**2 + y_particles**2))
    displacement_1 = np.append(displacement_1, MSD_1)
    displacement_2 = np.append(displacement_2, MSD_2)


D = 0.5
d_1= 1
d_2 = 2
analytical_1 =  2*d_1 * D *t_array
analytical_2 =  2*d_2 * D *t_array

plt.figure()
plt.title("Mean Square Displacement")
plt.ylabel("MSD")
plt.xlabel("time")
plt.plot(t_array, displacement_1, label = "simulation_1D")
plt.plot(t_array, analytical_1, label = "analytical_1D")
plt.grid()
plt.plot(t_array, displacement_2, label = "simulation_2D")
plt.plot(t_array, analytical_2, label = "analytical_2D")
plt.legend()
plt.show()
