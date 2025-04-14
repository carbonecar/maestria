import numpy as np
import matplotlib.pyplot as plt

#1. Dados los tiempos t= (0,1,3,4) y las mediciones b= (0,8,8,20).

# a)  Encontrar la recta y= c+ dt que mejor se ajusta a los datos. Para esto escriba las ecuaciones
#normales y resuélvalas. Calcular el error.

tiempos= np.array([0, 1, 3, 4])
mediciones= np.array([1, 8, 8, 20])

## graficamos
plt.plot(tiempos, mediciones, 'o', label='Datos')
# Se define la matriz de diseño
A= np.vstack((np.ones(len(tiempos)), tiempos)).T

print(A)

# Se resuelve el sistema de ecuaciones normales
# A.T @ A @ x = A.T @ b
solution=np.linalg.solve(A.T @ A, A.T @ mediciones)

print(solution)

