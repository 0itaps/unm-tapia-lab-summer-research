import numpy as np
import gpytoolbox
import matplotlib.pyplot as plt

def make_circle(num_points=32, radius=1.0, center=(0.0, 0.0)):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack((x, y))

square = np.array([
    [ 1.0, -1.0],
    [ 1.0,  1.0],
    [-1.0,  1.0],
    [-1.0, -1.0]
])

circle = make_circle(radius=1.2)

x_vals = np.linspace(-2, 2, 200)
y_vals = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x_vals, y_vals)
grid_points = np.column_stack((X.ravel(), Y.ravel()))

square_SDF = gpytoolbox.signed_distance(grid_points, square)[0]
circle_SDF = gpytoolbox.signed_distance(grid_points, circle)[0]

union_SDF = np.minimum(square_SDF, circle_SDF)
SDF_grid = union_SDF.reshape(X.shape)

abs_max = np.max(np.abs(SDF_grid))

plt.imshow(SDF_grid, extent=(-2, 2, -2, 2), origin='lower',
           cmap='RdBu_r', vmin=-abs_max, vmax=abs_max, interpolation='none')
plt.colorbar(label='Signed Distance')
plt.contour(X, Y, SDF_grid, levels=[0], colors='black')
plt.title('Signed Distance Field (Union of Square and Circle)')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.show()
