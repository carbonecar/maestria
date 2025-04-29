import numpy as np
import matplotlib.pyplot as plt

# Hiperparámetros
a = 1
b = 0.1
learning_rate = 0.1
num_epochs = 50
batch_size = 10

# Datos simulados (ej: para una red o regresión)
np.random.seed(42)
num_data_points = 100
X_data = np.random.randn(num_data_points, 2)

# Función de pérdida
def loss(x, data_batch):
    return np.mean([
        0.5 * (a * (x[0] - d[0])**2 + b * (x[1] - d[1])**2)
        for d in data_batch
    ])

# Gradiente del mini-batch
def grad(x, data_batch):
    gx = np.mean([a * (x[0] - d[0]) for d in data_batch])
    gy = np.mean([b * (x[1] - d[1]) for d in data_batch])
    return np.array([gx, gy])

# Inicialización
x = np.array([2.0, 2.0])
trajectory = [x.copy()]
losses = [loss(x, X_data)]

# Mini-batch loop
for epoch in range(num_epochs):
    np.random.shuffle(X_data)
    for i in range(0, num_data_points, batch_size):
        mini_batch = X_data[i:i+batch_size]
        g = grad(x, mini_batch)
        x -= learning_rate * g
    trajectory.append(x.copy())
    losses.append(loss(x, X_data))

# Graficar trayectoria
trajectory = np.array(trajectory)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o')
plt.title("Trayectoria de Mini-Batch GD")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(losses, marker='o')
plt.title("Pérdida promedio por época")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.yscale("log")
plt.grid(True)

plt.tight_layout()
plt.show()
