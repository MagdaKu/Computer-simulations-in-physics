import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import imageio.v2 as imageio 
import os
from tqdm import tqdm


def periodic_boundary(checked_element, L):
    return [(checked_element[0] + L) % L, (checked_element[1] + L) % L]

def check_element(lattice, checked_element, L, p):
    checked_element = periodic_boundary(checked_element, L)
    if lattice[checked_element[0], checked_element[1]] == -1:
        pc = np.random.random() < p
        if pc:
            cluster.append(checked_element)
            lattice[checked_element[0], checked_element[1]] = 1
        else:
            lattice[checked_element[0], checked_element[1]] = 0 
    return lattice

L = 100
p = 0.59
lattice = np.ones((L, L)) * (-1) #unchecked nodes
row = np.ones((1,L)) * (-2) #closed boundary
lattice = np.concatenate((row, lattice, row), axis = 0)
lattice[1,:] = 1 
cluster = deque(np.argwhere(lattice == 1)) # create the queue

file_path = ...#PATH
image_counter = 0 #used for animation
j = 0 #used for animation


#Task 1 

while len(cluster) != 0:
    image_counter +=1
    element = cluster[0]
    #upper element
    checked_element = [element[0]+1, element[1]]
    lattice = check_element(lattice, checked_element, L, p)
    #right element
    checked_element = [element[0], element[1]+1]
    lattice = check_element(lattice, checked_element, L, p)
    #lower element
    checked_element = [element[0]-1, element[1]]
    lattice = check_element(lattice, checked_element, L, p)
    #left element
    checked_element = [element[0], element[1]-1]
    lattice = check_element(lattice, checked_element, L, p)

    cluster.popleft()

    #section to save plots for animation
    '''
    if image_counter % L == 0:
        plt.imshow(lattice, interpolation="nearest", cmap = "magma")
        plt.title("Leath algorithm")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(file_path + f'img{j:06d}.png')
        j+= 1
    '''

#creating animation 
'''
filenames = sorted(os.listdir(file_path))
with imageio.get_writer(file_path + 'animation.gif', mode='I', duration = 2) as writer:
    for filename in filenames:
        if "img" in filename:
            image = imageio.imread(file_path + filename)
            writer.append_data(image)
'''




#Task 2
#cluster size
size_percolation  = 100
iterations = 100
p_array = np.linspace(0.5, 0.7, size_percolation)

plt.figure()
for L in [20,50, 100]:
    cluster_size = np.zeros(size_percolation)
    for p in tqdm(p_array):
        for k in range(iterations):
            unchecked = True
            lattice = np.ones((L, L)) * (-1)
            row = np.ones((1,L)) * (-2)
            lattice = np.concatenate((row, lattice, row), axis = 0)
            lattice[1,:] = 1 #Implement a queue
            cluster = deque(np.argwhere(lattice == 1)) # create the queue
            while len(cluster) != 0:
                element = cluster[0]
                #upper element
                checked_element = [element[0]+1, element[1]]
                if checked_element[0] == L and unchecked:
                    unchecked = False #we don't want to check clusters that percolate the whole grid
                lattice = check_element(lattice, checked_element, L, p)
                #right element
                checked_element = [element[0], element[1]+1]
                lattice = check_element(lattice, checked_element, L, p)
                #lower element
                checked_element = [element[0]-1, element[1]]
                lattice = check_element(lattice, checked_element, L, p)
                #left element
                checked_element = [element[0], element[1]-1]
                lattice = check_element(lattice, checked_element, L, p)

                cluster.popleft()

            if unchecked == True:
                cluster_size[np.where(p_array == p)[0]] += np.sum(lattice == 1)/L**2
                
    cluster_size = cluster_size/iterations
    plt.plot(p_array, cluster_size, label = f"L = {L}")


plt.ylabel("average cluster size S/L**2")
plt.xlabel("concentration")
plt.legend()
plt.show()


#percolation
size_percolation  = 100
iterations = 100
p_array = np.linspace(0.5, 0.7, size_percolation)
percolation = np.zeros(size_percolation)

plt.figure()
for L in [20,50, 100]:
    percolation = np.zeros(size_percolation)
    for p in tqdm(p_array):
        for k in range(iterations):
            unchecked = True
            lattice = np.ones((L, L)) * (-1)
            row = np.ones((1,L)) * (-2)
            lattice = np.concatenate((row, lattice, row), axis = 0)
            lattice[1,:] = 1 #Implement a queue
            cluster = deque(np.argwhere(lattice == 1)) # create the queue
            i = 0
            while len(cluster) != 0:
                element = cluster[i]
                #upper element
                checked_element = [element[0]+1, element[1]]
                if checked_element[0] == L and unchecked:
                    percolation[np.where(p_array == p)[0]] += 1
                    unchecked = False
                lattice = check_element(lattice, checked_element, L, p)
                #right element
                checked_element = [element[0], element[1]+1]
                lattice = check_element(lattice, checked_element, L, p)
                #lower element
                checked_element = [element[0]-1, element[1]]
                lattice = check_element(lattice, checked_element, L, p)
                #left element
                checked_element = [element[0], element[1]-1]
                lattice = check_element(lattice, checked_element, L, p)

                cluster.popleft()

    percolation = percolation/iterations
    plt.plot(p_array, percolation, label = f"L = {L}")

plt.title("Percolation probability")
plt.ylabel("probability")
plt.xlabel("concentration")
plt.legend()
plt.show()
