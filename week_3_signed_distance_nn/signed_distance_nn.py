import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gpytoolbox
import matplotlib.pyplot as plt

from week_2_signed_distances.signed_distance_union import square, circle

# 1. Data
def generate_sdf_dataset(shape_vertices, shape_id):
    x_vals = np.linspace(-2, 2, 200)
    y_vals = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    sdf = gpytoolbox.signed_distance(grid_points, shape_vertices)[0]

    N = grid_points.shape[0]
    shape_ids = np.full((N, 1), shape_id)
    inputs = np.hstack((grid_points, shape_ids))

    labels = sdf.reshape(-1, 1)

    return inputs, labels

inputs_square, labels_square = generate_sdf_dataset(square, 0)
inputs_circle, labels_circle = generate_sdf_dataset(circle, 1)

combined_inputs = np.concatenate((inputs_square, inputs_circle), axis=0)
combined_labels = np.concatenate((labels_square, labels_circle), axis=0)

full_data = np.hstack((combined_inputs, combined_labels))
np.random.shuffle(full_data)

inputs = full_data[:, :3]
labels = full_data[:, 3:]

inputs_tensor = torch.from_numpy(inputs).float()
labels_tensor = torch.from_numpy(labels).float()

#2. Model
class SignedDistanceNN(nn.Module):
    def __init__(self):
        super(SignedDistanceNN, self).__init__()
        self.hidden = nn.Linear(3,4)
        self.output = nn.Linear(4,1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

model = SignedDistanceNN()

#3. Loss & Optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

#4. Training Loop
for epoch in range(10000):
    pred = model(inputs_tensor)
    loss = loss_fn(pred, labels_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

#5. Test
with torch.no_grad():
    preds = model(inputs_tensor).cpu().numpy()

# Extract x, y, true sdf, predicted sdf
x = inputs[:, 0]
y = inputs[:, 1]
true_sdf = labels[:, 0]
pred_sdf = preds[:, 0]

# Plot True vs Predicted Signed Distances
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x, y, c=true_sdf, cmap='coolwarm', s=1)
plt.title('True Signed Distance')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(x, y, c=pred_sdf, cmap='coolwarm', s=1)
plt.title('Predicted Signed Distance')
plt.colorbar()

plt.show()


