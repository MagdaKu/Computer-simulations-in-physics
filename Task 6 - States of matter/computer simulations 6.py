import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio.v2 as imageio 
import os
from tqdm import tqdm

class Particle:
    def __init__(self, radius, position, velocity):
        self.rad = radius
        self.r = position
        self.v = velocity


def periodic_boundary(position, box_size):
    return (position + box_size) % box_size

def closest_image(x1, x2, box_size):
    x12 = x2 - x1
    x12 -= np.round(x12 / box_size) * box_size
    return x12


def potential(r,epsilon, sigma):
    return 4*epsilon * ((sigma/r)**12 - (sigma/r)**6)
    


def compute_velocities_and_positions(particles, box_size, eps, sigma):
    v_total = 0
    for p in particles:
        total_force = np.array([0.0, 0.0])
        for j in particles:
            if p==j:
                continue
            distance = closest_image(p.r, j.r, box_size)
            r = np.linalg.norm(distance)
            if r < 1e-12 or r > 2.5 * sigma:
                continue
            F = -48 * eps / sigma**2 * ((sigma / r)**14 - 0.5 * (sigma / r)**8) * distance / r
            total_force += F

        v0 = p.v
        v_new = v0 + total_force/m *dt/2 
        v_total += np.dot(v_new, v_new)*m

    temp_inst =  v_total/(2*particle_number*kB)
    eta = (temp/temp_inst)**0.5

    for p in particles:
        total_force = np.array([0.0, 0.0])
        for j in particles:
            if p==j:
                continue
            distance = closest_image(p.r, j.r, box_size)
            r = np.linalg.norm(distance)
            if r < 1e-12 or r > 2.5 * sigma:
                continue
            F = -48 * eps / sigma**2 * ((sigma / r)**14 - 0.5 * (sigma / r)**8) * distance / r
            total_force += F

        v1 = (2*eta - 1)* p.v + eta*total_force/m *dt
        v_new = (v1 + p.v)/2
        r_new = p.r + v1*dt 
        r_new = periodic_boundary(r_new, box_size)
        p.r = r_new
        p.v = v_new
    return p.r, p.v


def compute_energies_and_pressure(particles, particle_number, box_size, eps, sigma, kB):
    total_kinetic_energy = 0.0
    total_potential_energy = 0.0
    pressure_sum = 0.0
    potential_rc = potential(r = 2.5*sigma, epsilon = eps, sigma = sigma )
    for i, p1 in enumerate(particles):
        for j in range(i + 1, particle_number):  # Only iterate over each pair once
            p2 = particles[j]
            distance = closest_image(p1.r, p2.r, box_size)
            r = np.linalg.norm(distance)
            if r < 1e-12 or r > 2.5 * sigma:
                continue
            
            potential_energy = potential(r, eps, sigma) - potential_rc
            total_potential_energy += potential_energy
            force_magnitude = -48 * eps / sigma**2 * ((sigma / r)**14 - 0.5 * (sigma / r)**8)
            pressure_sum += r * force_magnitude

        total_kinetic_energy += 0.5 * m * np.dot(p1.v, p1.v)

    temperature = (2 * total_kinetic_energy) / (2* particle_number * kB)
    pressure = (particle_number * kB * temperature) / (box_size**2) + pressure_sum / (2 * box_size**2)
    return total_kinetic_energy, total_potential_energy, temperature, pressure


#parameters
particle_number = 16
box_size = 8.0
eps = 1.0
sigma = 1.0
radius = sigma / 2
dt = 0.025
temp = 0.1 #0.7 for liquid, 0.1 for solid
kB = 1
m = 1


nx = 4
ny = 4
dx = box_size/nx
dy= box_size/ny
particles = []

#Initializing the particles
for i in range(nx):
    for j in range(ny):
        position = np.array([i * dx + 1, j * dx + 1])
        velocity = np.array([(np.random.random() - 1 / 2), (np.random.random() - 1 / 2)])
        particles.append(Particle(radius, position, velocity))

sum_v = 0.0
for p in particles:
    sum_v += p.v
sum_v = sum_v / particle_number # center of mass velocity
for p in particles:
    p.v = p.v - sum_v # makes the center of mass stationary

sum_v2 = 0.0
for p in particles:
    sum_v2 += np.dot(p.v, p.v) / 2.0
sum_v2 = sum_v2 / particle_number # average kinetic Energy (assuming unit mass)
fs = np.sqrt(temp / sum_v2) # scaling factor
for p in particles:
    p.v = p.v * fs 


file_path = ... #PATH
N = 10000
total_energy = np.array([])
temperature_list = np.array([])
pressure_list = np.array([])


for i in tqdm(range(N)):
    p.r, p.v = compute_velocities_and_positions(particles, box_size, eps, sigma)
    kinetic_energy, potential_energy, temperature, pressure = compute_energies_and_pressure(particles, particle_number, box_size, eps, sigma, kB)
    total_energy = np.append(total_energy, kinetic_energy+potential_energy)   
    temperature_list = np.append(temperature_list, temperature)
    pressure_list = np.append(pressure_list, pressure)

    #section to make animation 
    '''
    if i%100 == 0:
        plt.clf() # clear the figure
        fig = plt.gcf() # define new figure
        for p in particles: 
            a = plt.gca()
            cir = Circle((p.r[0], p.r[1]), radius = p.rad) # draw circle in particle’s position
            a.add_patch(cir) 
            plt.plot()
        plt.xlim((0, box_size)) 
        plt.ylim((0, box_size))
        fig.set_size_inches((6, 6)) 
        plt.title(f'Lennard-Jones gas simulation, iteration {i:06d}')
        plt.savefig(file_path + f'img{i:06d}.png')
    '''


os_x = np.arange(N)
plt.figure()
plt.title("Energies and temperature")
plt.plot(os_x, total_energy, label = "Total energy")
plt.plot(os_x, temperature_list, label = "Temperature")
plt.plot(os_x, pressure_list, label = "Pressure")
plt.legend()
plt.tight_layout()
plt.show()


'''
#creating animation
filenames = sorted(os.listdir(file_path))

with imageio.get_writer(file_path + 'animation.gif', mode='I', duration = 0.7) as writer:
    for filename in filenames:
        image = imageio.imread(file_path + filename)
        writer.append_data(image)
'''