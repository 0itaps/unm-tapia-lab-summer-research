import numpy as np
import matplotlib.pyplot as plt

L1 = 1.0
L2 = 0.8
theta1 = 90
theta2 = 45

x1 = L1 * np.cos(np.radians(theta1))
y1 = L1 * np.sin(np.radians(theta1))

x2 = x1 + (L2 * np.cos(np.radians(theta1 + theta2)))
y2 = y1 + (L2 * np.sin(np.radians(theta1 + theta2)))

print(f"Coordinates: ({x1}, {y1})")
print(f"Coordinates: ({x2}, {y2})")

x_coords = np.array([0,x1,x2])
y_coords = np.array([0, y1, y2])

plt.plot(x_coords, y_coords, 'o-')
plt.axis('equal')
plt.show()
