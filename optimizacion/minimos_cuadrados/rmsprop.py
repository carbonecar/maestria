#Excelente, RMSProp es una mejora de AdaGrad que mantiene una media móvil del cuadrado de los gradientes 
# (en lugar de acumularlos indefinidamente). Esto evita que el learning rate se vuelva demasiado
# pequeño con el tiempo.
import numpy as np
import matplotlib.pyplot as plt

# Hiperparámetros
a = 1
b = 0.1
initial_lr = 0.1
rho = 0.9         # tasa de decaimiento exponencial (media móvil)
epsilon = 1e-8
num_epochs = 50
batch_size = 10

# Datos simulados
np.random.seed(42)
num_data_points = 100
X_data = np.random.randn(num_data_points, 2)

# Función de pérdida (MSE-like)
def loss(x, data_batch):
    return np.mean([
        0.5 * (a * (x[0] - d[0])**2 + b * (x[1] - d[1])**2)
        for d in data_batch
    ])

# Gradiente de la función de pérdida respecto a x
def grad(x, data_batch):
    gx = np.mean([a * (x[0] - d[0]) for d in data_batch])
    gy = np.mean([b * (x[1] - d[1]) for d in data_batch])
    return np.array([gx, gy])

# Inicialización
x = np.array([2.0, 2.0])
grad_squared_avg = np.zeros_like(x)  # media móvil de los cuadrados del gradiente

trajectory = [x.copy()]
losses = [loss(x, X_data)]

# Entrenamiento con RMSProp
for epoch in range(num_epochs):
    np.random.shuffle(X_data)
    for i in range(0, num_data_points, batch_size):
        batch = X_data[i:i+batch_size]

        g = grad(x, batch)

        grad_squared_avg = rho * grad_squared_avg + (1 - rho) * g**2

        adjusted_lr = initial_lr / (np.sqrt(grad_squared_avg) + epsilon)

        x -= adjusted_lr * g

    trajectory.append(x.copy())
    losses.append(loss(x, X_data))

# Gráficos
trajectory = np.array(trajectory)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o')
plt.title("Trayectoria con RMSProp")
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
