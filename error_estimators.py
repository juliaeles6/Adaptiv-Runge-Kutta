import importlib
import numpy as np
import RK 

importlib.reload(RK)

# Richardson estimator
def Richardson_estim(f, h, y_n, t_n, p, s, A, b, c):
    y1 = RK.RK_step(f, h, y_n, t_n, s, A, b, c)
    y_2_aux = RK.RK_step(f, h / 2, y_n, t_n, s, A, b, c)
    y2 = RK.RK_step(f, h / 2, y_2_aux, t_n + h / 2, s, A, b, c)

    error = np.linalg.norm(y2 - y1, ord=np.inf) / (2 ** p - 1)

    return y1, y2, error

# Embedded Runge-Kutta method estimator
def embedded_RK(f, h, y_n, t_n, s, A, b1, b2, c):
    y1 = RK.RK_step(f, h, y_n, t_n, s, A, b1, c)
    y2 = RK.RK_step(f, h, y_n, t_n, s, A, b2, c)

    error = np.linalg.norm(y2 - y1, ord=np.inf) 
    # print(f"y1={y1}, y2={y2}, error={error}")

    return y1, y2, error