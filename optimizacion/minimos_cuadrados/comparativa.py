 #Adam vs SGD vs RMSProp en un mismo pl

import numpy as np
import matplotlib.pyplot as plt

# Simulación para comparar Adam, SGD y RMSProp
a = 1
b = 0.1
lr = 0.1
num_epochs = 50
batch_size = 10
np.random.seed(42)

# Dataset simulado
num_data_points = 100
X_data = np.random.randn(num_data_points, 2)

def loss(x, data_batch):
    return np.mean([
        0.5 * (a * (x[0] - d[0])**2 + b * (x[1] - d[1])**2)
        for d in data_batch
    ])

def grad(x, data_batch):
    gx = np.mean([a * (x[0] - d[0]) for d in data_batch])
    gy = np.mean([b * (x[1] - d[1]) for d in data_batch])
    return np.array([gx, gy])

# Inicialización de trayectorias
def run_optimizer(opt_name):
    x = np.array([2.0, 2.0])
    losses = [loss(x, X_data)]
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0

    for epoch in range(num_epochs):
        np.random.shuffle(X_data)
        for i in range(0, num_data_points, batch_size):
            t += 1
            batch = X_data[i:i+batch_size]
            g = grad(x, batch)

            if opt_name == "SGD":
                x -= lr * g

            elif opt_name == "RMSProp":
                beta = 0.9
                v = beta * v + (1 - beta) * g**2
                x -= lr * g / (np.sqrt(v) + 1e-8)

            elif opt_name == "Adam":
                beta1 = 0.9
                beta2 = 0.999
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g**2
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                x -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        losses.append(loss(x, X_data))
    return losses

# Ejecutamos cada optimizador
losses_sgd = run_optimizer("SGD")
losses_rmsprop = run_optimizer("RMSProp")
losses_adam = run_optimizer("Adam")

# Gráfico comparativo
plt.figure(figsize=(10, 6))
plt.plot(losses_sgd, label="SGD", marker='o')
plt.plot(losses_rmsprop, label="RMSProp", marker='s')
plt.plot(losses_adam, label="Adam", marker='^')
plt.yscale("log")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Comparación de optimizadores (escala logarítmica)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
