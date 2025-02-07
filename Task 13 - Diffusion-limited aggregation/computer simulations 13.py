import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from tqdm import tqdm
from numba import jit


@jit(nopython=True)
def check_neighbours(x, y, structure):
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if (dx, dy) != (0, 0) and (x + dx, y + dy) in structure:
                return True
    return False

file_path = ...#PATH
grid_size = 501
grid = np.zeros((grid_size, grid_size))
centre = grid_size // 2

structure = {(centre, centre)}
grid[centre, centre] = 1
N_particles = 10000
R_max = grid_size // 2
particles = np.array([(centre + int(np.random.randint(0, R_max) * np.cos(theta)),
                        centre + int(np.random.randint(0, R_max) * np.sin(theta)))
                      for theta in np.random.uniform(0, 2 * np.pi, N_particles)], dtype=np.int32)

for x, y in particles:
    if 0 <= x < grid_size and 0 <= y < grid_size:
        grid[x, y] = 0.5

N = 10000
for i in tqdm(range(N)):
    new_particles = []
    for x, y in particles:
        if not check_neighbours(x, y, structure):
            x_move, y_move = np.random.choice([-1, 1]), np.random.choice([-1, 1])
            grid[x, y] = 0
            x += x_move
            y += y_move
            
            if 0 <= x < grid_size and 0 <= y < grid_size:
                grid[x, y] = 0.5
                new_particles.append((x, y))
        else:
            structure.add((x, y))
    
    particles = np.array(new_particles, dtype=np.int32)
    
    if i % 100 == 0:
        for x, y in structure:
            grid[x, y] = 1
        plt.imshow(grid, cmap="Greys")
        plt.savefig(os.path.join(file_path, f'img{i:04d}.png'))
    
    if len(particles) == 0:
        break


filenames = sorted([f for f in os.listdir(file_path) if f.endswith(".png")])
with imageio.get_writer(os.path.join(file_path, 'animation.gif'), mode='I', duration=1) as writer:
    for filename in filenames:
        writer.append_data(imageio.imread(os.path.join(file_path, filename)))