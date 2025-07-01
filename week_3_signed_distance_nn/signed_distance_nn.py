import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gpytoolbox
import matplotlib.pyplot as plt

from week_2_signed_distances.signed_distance_union import square, circle

# 1. Data
def generate_sdf_dataset(shape_vertices, shape_id):
    x_vals = np.linspace(-1.5, 1.5, 200)
    y_vals = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    sdf = gpytoolbox.signed_distance(grid_points, shape_vertices)[0]

    N = grid_points.shape[0]
    shape_ids = np.full((N, 1), shape_id)
    inputs = np.hstack((grid_points, shape_ids))
    labels = sdf.reshape(-1, 1)

    return inputs, labels

square_inputs, square_labels = generate_sdf_dataset(square, 0)
circle_inputs, circle_labels = generate_sdf_dataset(circle, 1)

combined_inputs = np.concatenate((square_inputs, circle_inputs), axis=0)
combined_labels = np.concatenate((square_labels, circle_labels), axis=0)

full_data = np.hstack((combined_inputs, combined_labels))
np.random.shuffle(full_data)

inputs = full_data[:, :3]
labels = full_data[:, 3:]

inputs_tensor = torch.from_numpy(inputs).float()
labels_tensor = torch.from_numpy(labels).float()

# 2. Model
class SignedDistanceNN(nn.Module):
    def __init__(self):
        super(SignedDistanceNN, self).__init__()
        self.hidden = nn.Linear(3, 64)
        self.output = nn.Linear(64, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

model = SignedDistanceNN()

# 3. Loss & Optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Training Loop
loss_history = []

for epoch in range(2000):
    pred = model(inputs_tensor)
    loss = loss_fn(pred, labels_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 5. Loss Plot
epochs = np.arange(len(loss_history))
loss_values = np.array(loss_history)

plt.figure()
plt.plot(epochs, loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss Over Time')
plt.grid(True)
plt.legend()
plt.show()

# 6. True SDF Visualizations
x_sq = square_inputs[:, 0]
y_sq = square_inputs[:, 1]
sdf_sq = square_labels[:, 0]

x_circ = circle_inputs[:, 0]
y_circ = circle_inputs[:, 1]
sdf_circ = circle_labels[:, 0]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x_sq, y_sq, c=sdf_sq, cmap='coolwarm', s=1)
plt.title("True SDF - Square")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(x_circ, y_circ, c=sdf_circ, cmap='coolwarm', s=1)
plt.title("True SDF - Circle")
plt.colorbar()

plt.tight_layout()
plt.show()

# 7. Predicted SDF Visualizations
square_tensor = torch.from_numpy(square_inputs).float()
circle_tensor = torch.from_numpy(circle_inputs).float()

with torch.no_grad():
    pred_sq = model(square_tensor).cpu().numpy().squeeze()
    pred_circ = model(circle_tensor).cpu().numpy().squeeze()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x_sq, y_sq, c=pred_sq, cmap='coolwarm', s=1)
plt.title("Predicted SDF - Square")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(x_circ, y_circ, c=pred_circ, cmap='coolwarm', s=1)
plt.title("Predicted SDF - Circle")
plt.colorbar()

plt.tight_layout()
plt.show()
