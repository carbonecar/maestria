import numpy as np
import matplotlib.pyplot as plt

# Parámetros
a = 1
b = 0.1
numero_pasos = 100
learning_rate = 0.1

# Hiperparámetros de Adam
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Gradiente de la función
def evaluar_gradiente(vector):
    x, y = vector
    return np.array([a * x, b * y])

# Inicialización
xk = np.zeros((numero_pasos + 1, 2))
xk[0] = [1.0, 1.0]  # punto inicial

mt = np.zeros(2)
vt = np.zeros(2)

# Historial del valor de la función
f_hist = []

for t in range(1, numero_pasos + 1):
    grad = evaluar_gradiente(xk[t-1])

    # Actualizar momentos
    mt = beta1 * mt + (1 - beta1) * grad
    vt = beta2 * vt + (1 - beta2) * (grad ** 2)

    # Corrección de sesgo
    mt_hat = mt / (1 - beta1 ** t)
    vt_hat = vt / (1 - beta2 ** t)

    # Actualizar posición
    xk[t] = xk[t-1] - learning_rate * mt_hat / (np.sqrt(vt_hat) + epsilon)

    # Guardar valor de f
    f_val = 0.5 * (a * xk[t][0]**2 + b * xk[t][1]**2)
    f_hist.append(f_val)

# Visualización
plt.plot(f_hist)
plt.title("Convergencia de Adam")
plt.xlabel("Paso")
plt.ylabel("f(x, y)")
plt.grid(True)
plt.show()
