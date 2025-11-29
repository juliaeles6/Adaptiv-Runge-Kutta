import importlib
import numpy as np
import error_estimators as EE
import controllers as CS
import nodepy.runge_kutta_method as rk # type: ignore
importlib.reload(EE)


# Adaptive Runge-Kutta method with Richardson error estimation
def adaptive_Richardson_RK(f, y_0, t_0, t_f, h_0, TOL, rho_min, rho_max, m, s, A, b, c):
    myMethod = rk.RungeKuttaMethod(A, b, c, name = "MyMethod")
    p = myMethod.order()

    h = h_0
    h_s = [h]
    f_ = lambda t, y: np.asarray(f(t, y))
    y = [y_0]
    t = [t_0]
    
    n = 0
    while t[n] < t_f:
        y1, y2, error = EE.Richardson_estim(f_, h, y[n], t[n], p, s, A, b, c)

        if error <= TOL:
            t.append(t[n] + h) 
            y.append(y1)
            n += 1
            h_s.append(h)
            continue

        rho = m * (TOL / error) ** (1 / (p + 1))
        h *= min(rho_max, max(rho_min, rho))        

    return h_s, y, t

# Adaptive Runge-Kutta method with embedded RK
def embedded_adaptive_RK(f, y_0, t_0, t_f, h_0, TOL, rho_min, rho_max, m, s, A, b1, b2, c):
    myMethod = rk.RungeKuttaMethod(A, b1, c, name = "MyMethod")
    myEmbeddedMethod = rk.RungeKuttaMethod(A, b2, c, name = "MyEmbeddedMethos")
    p = myMethod.order()
    pe = myEmbeddedMethod.order()

    h = h_0
    h_s = [h]
    f_ = lambda t, y: np.asarray(f(t, y))
    t = [t_0]
    y = [y_0]

    n = 0
    print(h)
    while t[n] < t_f:
        y1, y2, error = EE.embedded_RK(f_, h, y[n], t[n], s, A, b1, b2, c)

        # print(f"error={error}, h={h}")
        if error <= TOL:
            t.append(t[n] + h) 
            y.append(y1)
            n += 1
            h_s.append(h)
            # print(f"n={n}, error={error}, h={h}, y={y1}")
            continue

        rho = m * (TOL / error) ** (1 / (p + 1))
        h *= min(rho_max, max(rho_min, rho))        
        # print(f"error={error}, rho={rho} h={h}")

    # print(y)
    return h_s, y, t

# adaptive Runge-Kutta method with using controllers
def adaptive_RK_with_controller(f, y_0, t_0, t_f, h_0, TOL, controller, s, A, b1, b2, c):
    myMethod = rk.RungeKuttaMethod(A, b1, c, name = "MyMethod")
    myEmbeddedMethod = rk.RungeKuttaMethod(A, b2, c, name = "MyEmbeddedMethos")
    p = myMethod.order()
    pe = myEmbeddedMethod.order()

    h = h_0
    h_s = [h]
    f_ = lambda t, y: np.asarray(f(t, y))
    t = [t_0]
    y = [y_0]
    
    error_prev = error_prev2 = TOL

    n = 0
    while t[n] < t_f:
        y1, y2, error = EE.embedded_RK(f_, h, y[n], t[n], s, A, b1, b2, c)

        if error < TOL:
            t[n + 1] = t[n] + h
            y[n + 1] = y2

            n += 1
            error_prev2 = error_prev
            error_prev = error
            continue

        if controller == "I":
            S = CS.I_controller(error, TOL, p)
        elif controller == "PI":
            S = CS.PI_controller(error, error_prev, TOL, p) 
        elif controller == "PID":
            S = CS.PID_controller(error, error_prev, error_prev2, TOL, p)
        else:
            raise ValueError("Unknown controller")

        S = max(0.1, min(5.0, S))
        h *= S

    return h_s, y, t