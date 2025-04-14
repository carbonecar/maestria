import numpy as np

A = np.array([[4, 7], [2, 6]])
print(A)
A_inv = np.linalg.inv(A)
I=np.linalg.matmul(A, A_inv)
print(A_inv)
print(I)


# Calculo de la pseudoinversa
A = np.array([[1, 2], [3, 4], [5, 6]])  # Matriz no cuadrada
A_pinv = np.linalg.pinv(A)

print(A_pinv)
print(np.matmul(A, A_pinv))


