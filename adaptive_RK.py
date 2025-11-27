import numpy as np
import error_estimators as EE
import RK
import nodepy.runge_kutta_method as rk # type: ignore

# Adaptive Runge-Kutta method with Richardson error estimation
def adaptive_Richardson_RK(f, y_0, t_0, t_f, N, TOL, rho_min, rho_max, m, s, A, b, c):
    myMethod = rk.RungeKuttaMethod(A, b, c, name = "MyMethod")
    p = myMethod.order()

    h = (t_f - t_0) / N
    t = np.zeros(N+1)
    y = np.zeros(N+1)
    y[0] = y_0
    t[0] = t_0
    
    k = np.zeros(s)
    
    n = 0
    while n < N:
        y1, y2, error = EE.Richardson_estim(f, h, y[n], t[n], p, s, A, b, c)

        if error < TOL:
            t[n + 1] = t[n] + h 
            y[n + 1] = y2
            n += 1

        h *= np.min(rho_max, np.max(rho_min, m * (1 / error) ** (1 / (p + 1))))
    return h, y, t

# Adaptive Runge-Kutta method with embedded RK
def embedded_adaptive_RK(f, y_0, t_0, t_f, N, TOL, rho_min, rho_max, m, s, A, b1, b2, c):
    myMethod = rk.RungeKuttaMethod(A, b1, c, name = "MyMethod")
    myEmbeddedMethod = rk.RungeKuttaMethod(A, b2, c, name = "MyEmbeddedMethos")
    p = myMethod.order()
    pe = myEmbeddedMethod.order()

    h = (t_f - t_0) / N
    t = np.zeros(N+1)
    y = np.zeros(N+1)
    y[0] = y_0
    t[0] = t_0
    
    k = np.zeros(s)
    
    n = 0
    while n < N:
        y1, y2, error = EE.embedded_RK(f, h, y[n], t[n], s, A, b1, b2, c)

        if error < TOL:
            t[n + 1] = t[n] + h 
            y[n + 1] = y2
            n += 1

        h *= np.min(rho_max, np.max(rho_min, m * (1 / error) ** (1 / (p + 1))))
    
    return h, y, t