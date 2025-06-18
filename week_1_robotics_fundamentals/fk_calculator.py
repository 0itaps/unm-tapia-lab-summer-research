import numpy as np
import matplotlib.pyplot as plt

def fk_calculator(lengths, angles):
    positions = [(0,0)]
    x, y = 0, 0
    total_angle = 0

    for i in range(len(lengths)):
        total_angle += angles[i]
        x += lengths[i] * np.cos(total_angle)
        y += lengths[i] * np.sin(total_angle)
        positions.append((x,y))

    return np.array(positions)

def plot_robot(positions):
    x_vals, y_vals = positions[:, 0], positions[:, 1]
    plt.plot(x_vals, y_vals, 'o-')
    plt.axis('equal')
    plt.show()

lengths = [1, 1, 1]
angles = [np.pi/6, np.pi/4, -np.pi/6]

positions = fk_calculator(lengths, angles)
plot_robot(positions)
