#Perfecto. Adam (Adaptive Moment Estimation) combina lo mejor de momentum y 
# RMSProp: usa tanto la media móvil del gradiente (como momentum) 
# como la media móvil del cuadrado del gradiente (como RMSProp).


#📌 ¿Qué hace Adam?
#Usa la idea de momentum: suaviza la dirección de actualización con la media móvil del gradiente.

#Usa la idea de RMSProp: adapta el learning rate por dimensión con base en el segundo momento.

#Corrige los sesgos de los momentos para los primeros pasos.


import numpy as np
import matplotlib.pyplot as plt

# Hiperparámetros
a = 1
b = 0.1
lr = 0.1
beta1 = 0.9
beta2 = 0.999
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
m = np.zeros_like(x)  # primer momento (media de gradientes)
v = np.zeros_like(x)  # segundo momento (media de gradientes al cuadrado)

trajectory = [x.copy()]
losses = [loss(x, X_data)]

t = 0  # paso de tiempo

# Entrenamiento con Adam
for epoch in range(num_epochs):
    np.random.shuffle(X_data)
    for i in range(0, num_data_points, batch_size):
        t += 1
        batch = X_data[i:i+batch_size]
        g = grad(x, batch)

        # Actualizar momentos
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2

        # Corregir bias
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Actualizar parámetros
        x -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

    trajectory.append(x.copy())
    losses.append(loss(x, X_data))

# Gráficos
trajectory = np.array(trajectory)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o')
plt.title("Trayectoria con Adam")
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
