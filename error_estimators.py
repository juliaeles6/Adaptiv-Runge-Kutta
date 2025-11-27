import numpy as np
import RK 

# Richardson estimator
def Richardson_estim(f, h, y_n, t_n, p, s, A, b, c):
    y1 = RK.RK_step(f, h, y_n, t_n, s, A, b, c)
    y_2_aux = RK.RK_step(f, h / 2, y_n, t_n, s, A, b, c)
    y2 = RK.RK_step(f, h / 2, y_2_aux, t_n + h / 2, s, A, b, c)

    error = np.linalg.norm(y2 - y1, ord=np.inf) / (2 ** p - 1)

    return y1, y2, error

# Embedded Runge-Kutta method estimator
def embedded_RK(f, h, y_n, t_n, s, A, b1, b2, c):
    k = RK.get_k_values(f, h, y_n, t_n, s, A, c)

    y1 = np.dot(k, b1)
    y2 = np.dot(k, b2)

    error = np.linalg.norm(y2 - y1, ord = np.inf) 

    return y1, y2, error