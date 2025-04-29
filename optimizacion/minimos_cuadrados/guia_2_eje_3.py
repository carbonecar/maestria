import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def generate_matrix_with_condition_number(n, cond):
    """Genera una matriz aleatoria de tamaño n x n con un número de condición dado."""
    U = la.qr(np.random.rand(n, n))[0]  # Matriz ortogonal aleatoria
    V = la.qr(np.random.rand(n, n))[0]  # Otra matriz ortogonal aleatoria
    s = np.logspace(0, np.log10(cond), n)  # Valores singulares espaciados logarítmicamente
    S = np.diag(s)
    A = U @ S @ V.T
    return A

def exact_line_search_step(A, b, x_k, grad_fk):
    """Calcula el tamaño del paso óptimo (alpha) usando exact line search."""
    numerator = grad_fk.T @ grad_fk
    denominator = grad_fk.T @ A.T @ A @ grad_fk
    if denominator == 0:
        return 0  # Evitar división por cero si el gradiente está en el núcleo de A
    alpha = numerator / denominator
    return alpha

def gradient_descent_exact_linesearch(A, b, x_0, max_iter=1000, tol=1e-6):
    """Implementa el algoritmo de gradiente descendente con exact line search."""
    x_k = x_0
    history = [np.linalg.norm(A @ x_k - b)**2]
    for i in range(max_iter):
        residual = A @ x_k - b
        grad_fk = A.T @ residual
        alpha_k = exact_line_search_step(A, b, x_k, grad_fk)
        x_k_next = x_k - alpha_k * grad_fk
        norm_residual_squared = np.linalg.norm(A @ x_k_next - b)**2
        history.append(norm_residual_squared)
        if np.linalg.norm(x_k_next - x_k) < tol:
            break
        x_k = x_k_next
    return x_k, history

def pseudo_inverse_solution(A, b):
    """Calcula la solución del problema de mínimos cuadrados usando la pseudo-inversa."""
    A_pinv = la.pinv(A)
    x_pinv = A_pinv @ b
    return x_pinv, np.linalg.norm(A @ x_pinv - b)**2

def study_convergence(n=100, condition_numbers=[10, 100, 1000, 10000], num_runs=5):
    """Estudia la convergencia del GD en función del número de condición."""
    results = {}
    plt.figure(figsize=(12, 8))

    for cond in condition_numbers:
        convergence_histories = []
        final_residuals_gd = []

        for _ in range(num_runs):
            A = generate_matrix_with_condition_number(n, cond)
            b = np.random.rand(n)
            x_0 = np.random.rand(n)  # Punto inicial aleatorio

            x_gd, history_gd = gradient_descent_exact_linesearch(A, b, x_0, max_iter=5000, tol=1e-8)
            x_pinv, residual_pinv = pseudo_inverse_solution(A, b)
            final_residuals_gd.append(history_gd[-1])
            convergence_histories.append(history_gd)

        avg_convergence = np.mean(np.array([hist[:min(len(h) for h in convergence_histories)] for hist in convergence_histories]), axis=0)
        results[cond] = {'gd_residuals': final_residuals_gd, 'pinv_residual': residual_pinv, 'convergence': avg_convergence}

        plt.plot(avg_convergence, label=f'Cond(A) = {cond}')

    plt.xlabel('Iteraciones')
    plt.ylabel('Norma del Residuo al Cuadrado')
    plt.title('Convergencia del Descenso por Gradiente con Exact Line Search')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nResultados de la convergencia en función del número de condición:")
    for cond, data in results.items():
        print(f"\nNúmero de Condición: {cond}")
        print(f"  Residuales GD Finales (promedio): {np.mean(data['gd_residuals']):.4e}")
        print(f"  Residual Pseudo-inversa: {data['pinv_residual']:.4e}")

if __name__ == "__main__":
    n = 100
    cond_number = 1000  # Ejemplo de número de condición
    A = generate_matrix_with_condition_number(n, cond_number)
    b = np.random.rand(n)
    x_0 = np.random.rand(n)

    print("Resolviendo el problema de mínimos cuadrados con Descenso por Gradiente (Exact Line Search):")
    x_gd, history_gd = gradient_descent_exact_linesearch(A, b, x_0, max_iter=5000, tol=1e-8)
    residual_gd_final = np.linalg.norm(A @ x_gd - b)**2
    print(f"  Solución encontrada por GD: {x_gd[:5]}...")
    print(f"  Residual final (GD): {residual_gd_final:.4e}")
    print(f"  Número de iteraciones GD: {len(history_gd) - 1}")

    print("\nResolviendo el problema de mínimos cuadrados con la Pseudo-inversa:")
    x_pinv, residual_pinv = pseudo_inverse_solution(A, b)
    print(f"  Solución encontrada por Pseudo-inversa: {x_pinv[:5]}...")
    print(f"  Residual (Pseudo-inversa): {residual_pinv:.4e}")

    print("\nEstudiando la convergencia en función del número de condición:")
    study_convergence()



    