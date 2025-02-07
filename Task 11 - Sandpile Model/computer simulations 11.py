import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio.v2 as imageio 
import os


def toppling_a(grid, height_crit, N_iter):
    grains_per_cell = np.array([])
    for i in range(N_iter):
        #add a grain
        position = (np.random.randint(0, len(grid)), np.random.randint(0, len(grid)))
        grid[position]+= 1
        while np.max(grid) > height_crit:
            ix,iy = np.where(grid > height_crit)
            #topple
            grid[ix,iy] -= 4
            grid[ix+1, iy] += 1
            grid[ix-1,iy] += 1
            grid[ix,iy + 1] += 1
            grid[ix,iy-1] += 1
        grid[0] = 0
        grid[-1] = 0
        grid[:, 0] = 0
        grid[:, -1] = 0
        grains_per_cell = np.append(grains_per_cell, np.sum(grid) / (len(grid)-2)**2)
    return grid, grains_per_cell



def toppling_b(grid, height_crit, N_iter, file_path):
    for i in tqdm(range(N_iter)):
        #add a grain
        position = (len(grid)//2 + 1, len(grid)//2 + 1)
        grid[position]+= 1
        while np.max(grid) > height_crit:
            ix,iy = np.where(grid > height_crit)
            #topple
            grid[ix,iy] -= 4
            grid[ix+1, iy] += 1
            grid[ix-1,iy] += 1
            grid[ix,iy + 1] += 1
            grid[ix,iy-1] += 1
        grid[0] = 0
        grid[-1] = 0
        grid[:, 0] = 0
        grid[:, -1] = 0
        if i%100 == 0:
            plt.clf() # clear the figure
            fig = plt.gcf() # define new figure
            plt.imshow(grid, cmap='copper', interpolation='nearest')
            fig.set_size_inches((6, 6)) # figure size
            plt.title(f'Adding grains at the center of the system, iteration {i:06d}')
            plt.savefig(file_path + f'img{i:06d}.png')
    return grid


def toppling_c(grid, height_crit, N_iter):
    size_of_avalanche = np.array([])
    for i in tqdm(range(N_iter)):
        avalanche = np.zeros((len(grid), len(grid)))
        #add a grain
        position = (np.random.randint(0, len(grid)), np.random.randint(0, len(grid)))
        grid[position]+= 1
        while np.max(grid) > height_crit:
            ix,iy = np.where(grid > height_crit)
            #topple
            grid[ix,iy] -= 4
            grid[ix+1, iy] += 1
            grid[ix-1,iy] += 1
            grid[ix,iy + 1] += 1
            grid[ix,iy-1] += 1
            avalanche[ix,iy] = 1
        grid[0] = 0
        grid[-1] = 0
        grid[:, 0] = 0
        grid[:, -1] = 0
        size_of_avalanche = np.append(size_of_avalanche, np.sum(avalanche)) 
    return grid, size_of_avalanche




def toppling_d(grid, height_crit, N_iter, file_path):
    search_counter = 0
    
    for i in tqdm(range(N_iter)):
        # Toppling process
        while np.max(grid) > height_crit:
            ix, iy = np.where(grid > height_crit)
            search_counter += 1
            for x, y in zip(ix, iy):
                grid[x, y] -= 4  # Topple the current cell
                if x + 1 < grid.shape[0]: grid[x + 1, y] += 1  # Down
                if x - 1 >= 0: grid[x - 1, y] += 1  # Up
                if y + 1 < grid.shape[1]: grid[x, y + 1] += 1  # Right
                if y - 1 >= 0: grid[x, y - 1] += 1  # Left

            # Save plots every 100 iterations of toppling
            if search_counter % 100 == 0:
                plt.clf()  # Clear the figure
                fig = plt.gcf()  # Define a new figure
                plt.imshow(grid, cmap='copper', interpolation='nearest')
                fig.set_size_inches((6, 6))  # Set figure size
                plt.title(f'Supercritical grid, iteration {i:06d}')
                plt.savefig(file_path + f'img{i:06d}.png')

        # Clear the edges (grains topple off)
        grid[0, :] = 0
        grid[-1, :] = 0
        grid[:, 0] = 0
        grid[:, -1] = 0

    return grid


file_path = ...#PATH
height_crit = 3
grid = np.zeros((33,33))
N_iter = 10000
N_array = np.arange(N_iter)

#Task A
grid_a, grains_per_cell = toppling_a(grid, height_crit, N_iter)
plt.plot(N_array, grains_per_cell)
plt.title("The number of grains per cell in the system")
plt.xlabel("Steps")
plt.ylabel("Number of grains per cell")
plt.show()


#Task B

grid_b = toppling_b(grid, height_crit, N_iter, file_path)
#creating animation
filenames = sorted(os.listdir(file_path))
with imageio.get_writer(file_path + 'animation.gif', mode='I', duration = 0.7) as writer:
    for filename in filenames:
        image = imageio.imread(file_path + filename)
        writer.append_data(image)

#Task C
N_iter = 50000
grid_c, size_of_avalanche = toppling_c(grid, height_crit, N_iter)
size_of_avalanche = list(size_of_avalanche[1000:])
avmax = np.max(size_of_avalanche)
S = np.arange(1, avmax + 1)
a = S[0]
N_count = [size_of_avalanche.count(s) for s in S]
os = np.arange(1,701)
results = 1/os/10
h, bins = np.histogram(size_of_avalanche, bins = 100, density = True)

plt.xscale('log')
plt.yscale('log')
points = (bins[:-1] + bins[1:]/2)
plt.plot(points, h)
plt.plot(os, results)
plt.title("Number of avalanches as a function of size")
plt.show()

#Task D
grid = np.ones((52,52)) * 7 #supercritical grid
grid[0] = 0
grid[-1] = 0
grid[:, 0] = 0
grid[:, -1] = 0

N_iter = 15000

grid_d = toppling_d(grid, height_crit, N_iter, file_path)
#creating animation
filenames = sorted(os.listdir(file_path))
with imageio.get_writer(file_path + 'animation.gif', mode='I', duration = 0.7) as writer:
    for filename in filenames:
        image = imageio.imread(file_path + filename)
        writer.append_data(image)
