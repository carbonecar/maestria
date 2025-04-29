import numpy as np
import matplotlib.pyplot as plt

# Hiperparámetros
a = 1
b = 0.1
learning_rate = 0.1
num_epochs = 50

# Simulamos puntos de datos (ej: para una red o regresión)
# En este caso, son solo para simular distintos gradientes
np.random.seed(42)
num_data_points = 100
X_data = np.random.randn(num_data_points, 2)

# Función objetivo
def loss(x, data_point):
    return 0.5 * (a * (x[0] - data_point[0])**2 + b * (x[1] - data_point[1])**2)

# Gradiente estocástico de la función con respecto a un dato
def grad(x, data_point):
    return np.array([
        a * (x[0] - data_point[0]),
        b * (x[1] - data_point[1])
    ])

# Inicialización
x = np.array([2.0, 2.0])
trajectory = [x.copy()]
losses = [np.mean([loss(x, d) for d in X_data])]

# SGD loop
for epoch in range(num_epochs):
    np.random.shuffle(X_data)
    for data_point in X_data:
        g = grad(x, data_point)
        x -= learning_rate * g
    trajectory.append(x.copy())
    losses.append(np.mean([loss(x, d) for d in X_data]))

# Graficar trayectoria (x,y)
trajectory = np.array(trajectory)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o')
plt.title("Trayectoria de SGD")
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
