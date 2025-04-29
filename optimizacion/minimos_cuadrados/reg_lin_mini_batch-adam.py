import numpy as np
import matplotlib.pyplot as plt

# Datos sintéticos
np.random.seed(42)
n_samples = 1000
X = 2 * np.random.rand(n_samples, 1)
true_w = 3.5
true_b = -1.2
y = true_w * X + true_b + 0.5 * np.random.randn(n_samples, 1)

# Inicialización de parámetros
w = np.random.randn()
b = 0.0

# Hiperparámetros
lr = 0.05
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
epochs = 50
batch_size = 32

# Momentos iniciales
m_w, v_w = 0.0, 0.0
m_b, v_b = 0.0, 0.0

# Historia de pérdida
loss_hist = []

# Función de pérdida y gradientes
def compute_loss(X_batch, y_batch, w, b):
    preds = X_batch * w + b
    error = preds - y_batch
    loss = np.mean(error ** 2)
    grad_w = 2 * np.mean(error * X_batch)
    grad_b = 2 * np.mean(error)
    return loss, grad_w, grad_b

t = 0  # contador de pasos para bias correction

for epoch in range(epochs):
    # Mezclar los datos
    indices = np.random.permutation(n_samples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    for i in range(0, n_samples, batch_size):
        t += 1
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        
        loss, grad_w, grad_b = compute_loss(X_batch, y_batch, w, b)

        # Actualizar momentos
        m_w = beta1 * m_w + (1 - beta1) * grad_w
        v_w = beta2 * v_w + (1 - beta2) * grad_w ** 2
        m_b = beta1 * m_b + (1 - beta1) * grad_b
        v_b = beta2 * v_b + (1 - beta2) * grad_b ** 2

        # Corrección de sesgo
        m_w_hat = m_w / (1 - beta1 ** t)
        v_w_hat = v_w / (1 - beta2 ** t)
        m_b_hat = m_b / (1 - beta1 ** t)
        v_b_hat = v_b / (1 - beta2 ** t)

        # Actualizar parámetros
        w -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        b -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
        
        loss_hist.append(loss)

# Mostrar resultados
print(f"Peso aprendido: w = {w:.4f}, b = {b:.4f}")
plt.plot(loss_hist)
plt.title("Pérdida durante el entrenamiento con Adam (Mini-batch)")
plt.xlabel("Iteraciones")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
