import numpy as np
import matplotlib.pyplot as plt
import pygame
import time
import tkinter as tk
from tkinter import simpledialog
import os
import imageio


COLOR_BG = (255, 255, 255)
COLOR_GRID = (170, 170, 170)
COLOR_ALIVE = (10, 10, 10)


pygame.init()
pygame.display.set_caption("Conway's game of life")


def update(screen, cells, size, with_progress=False):
    updated_cells = np.zeros((cells.shape[0], cells.shape[1]))

    for row, col in np.ndindex(cells.shape):
        alive = np.sum(cells[row-1:row+2, col-1:col+2]) - cells[row, col]
        color = COLOR_BG if cells[row, col] == 0 else COLOR_ALIVE

        if cells[row, col] == 1:
            if alive < 2 or alive > 3:
                updated_cells[row, col] = 0
                if with_progress:
                    color = COLOR_BG
            elif 2 <= alive <= 3:
                updated_cells[row, col] = 1
                if with_progress:
                    color = COLOR_ALIVE
        else:
            if alive == 3:
                updated_cells[row, col] = 1
                if with_progress:
                    color = COLOR_ALIVE

        pygame.draw.rect(screen, color, (col * size, row * size, size - 1, size - 1))

    return updated_cells



def get_user_input():
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    cols = simpledialog.askinteger("Input", "Enter the number of columns:", minvalue=1, maxvalue=200)
    rows = simpledialog.askinteger("Input", "Enter the number of rows:", minvalue=1, maxvalue=200)
    animation = simpledialog.askstring("Input", "Would you like to save the animation? (y/n)")
    animation = animation.lower() == "y"
    root.destroy()  # Close the Tkinter window
    return rows, cols, animation


def save_frame(screen, frame_count):
    """Save the current screen as an image."""
    file_path = ...#PATH
    filename = os.path.join(file_path, f"frame_{frame_count:04d}.png")
    pygame.image.save(screen, filename)
    return filename

def create_animation(duration):
    """Create a GIF from saved frames."""
    file_path = ...#PATH
    filenames = sorted(os.listdir(file_path))  # Sort frame filenames
    output_file = file_path + 'game_of_life.gif'
    with imageio.get_writer(output_file, mode="I", duration=duration) as writer:
        for filename in filenames:
            if "frame" in filename:
                image = imageio.imread(os.path.join(file_path, filename))
                writer.append_data(image)
    print(f"Animation saved as {output_file}")

    filenames = sorted(os.listdir(file_path))


def main():
    pygame.init()
    rows, cols, animation = get_user_input()

    target_width, target_height = 800, 600  # Approximate default screen size
    cell_size_x = target_width // cols
    cell_size_y = target_height // rows
    cell_size = min(cell_size_x, cell_size_y)

    screen_width = cols * cell_size
    screen_height = rows * cell_size

    cells = np.zeros((rows, cols))
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Conway's Game of Life")

    screen.fill(COLOR_GRID)
    update(screen, cells, cell_size)
    pygame.display.flip()
    pygame.display.update()
    frame_count = 1
    
    max_frames = 150  # Maximum number of frames to save
    previous_cells = np.copy(cells)
    
    running = False
    first_image_saved = False

    while True:
        for Q in pygame.event.get():
            if Q.type == pygame.QUIT:
                pygame.quit()
                return
            elif Q.type == pygame.KEYDOWN:
                if Q.key == pygame.K_SPACE:
                    running = not running
                    update(screen, cells, cell_size)
                    pygame.display.update()
                    if not first_image_saved:
                        save_frame(screen, frame_count=frame_count)
                        frame_count += 1
                        first_image_saved = True

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                if cells[pos[1] // cell_size, pos[0] // cell_size] == 0:
                    cells[pos[1] // cell_size, pos[0] // cell_size] = 1
                else:
                    cells[pos[1] // cell_size, pos[0] // cell_size] = 0
                update(screen, cells, cell_size)
                pygame.display.update()

        screen.fill(COLOR_GRID)

        if running:
            cells = update(screen, cells, cell_size, with_progress=True)
            pygame.display.update()
            if animation:
                save_frame(screen, frame_count)
                frame_count += 1
                if frame_count >= max_frames or np.sum(cells) == 0 or np.array_equal(cells, previous_cells):
                    print(f"Saved {frame_count} frames. Stopping animation.")
                    create_animation(duration = 2)  # Create the GIF after saving frames
                    pygame.quit()
                    return
                previous_cells = np.copy(cells)
        time.sleep(0.05)

if __name__ == "__main__":
    main()






























'''
def live_or_die():
    pass

#Conway's game of life
N = 10
size = 10
file_path = f"C:/Users/Magda/Documents/Magda/Computer simulations/computer simulations 12 plots/" 
#grid = np.random.randint(2, size = (size,size))
grid = np.zeros((size, size))
#grid[1,5],grid[1,6], grid[2,4],grid[2,5], grid[2,6],grid[3,5], grid[3,6] = 1,1,1,1,1,1,1
grid[4,4],grid[4,6], grid[5,5],grid[5,6], grid[6,5 ] = 1,1,1,1,1
plt.imshow(grid, cmap = "Greys")
plt.savefig(file_path + f'img{0:03d}.png')
N_iter = 100

for k in range(N_iter):
    for i in range(size):
        for j in range(size):
            newGrid = grid.copy() 
            total = int((grid[i, (j-1)%N] + grid[i, (j+1)%N] + 
                         grid[(i-1)%N, j] + grid[(i+1)%N, j] + 
                         grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] + 
                         grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N])) 
            if grid[i,j] == 1:
                if total < 2 or total > 3:
                    newGrid[i,j] = 0
            if grid[i,j] == 0:
                if total ==3:
                    newGrid[i,j] = 1
            grid = newGrid        
    plt.imshow(grid, cmap = "Greys")
    plt.savefig(file_path + f'img{k+1:03d}.png')

'''

    