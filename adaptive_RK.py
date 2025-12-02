import importlib
import numpy as np
import warnings
import nodepy.runge_kutta_method as rk # type: ignore
import error_estimators as EE
import controllers as CS
importlib.reload(EE)

# Adaptive Runge-Kutta method with Richardson error estimation
def adaptive_Richardson_RK(f, y_0, t_0, t_f, h_0, TOL, rho_min, rho_max, m, s, A, b, c):
    myMethod = rk.ExplicitRungeKuttaMethod(A, b, c, name = "MyMethod")
    p = myMethod.order()

    cnt_accepted = 0
    cnt_rejected = 0

    h = h_0
    h_s = [h]
    f_ = lambda t, y: np.asarray(f(t, y))
    y = [y_0]
    t = [t_0]
    
    n = 0
    while t[n] < t_f:
        y1, y2, error = EE.Richardson_estim(f_, h, y[n], t[n], p, s, A, b, c)
        rho = rho_max if error == 0 else m * (TOL / error) ** (1 / (p + 1))

        if error <= TOL:
            if t[n] + h == t[n]:
                warnings.warn("Step size became too small; stopping to avoid infinite loop.")
                break
            cnt_accepted += 1
            t.append(t[n] + h) 
            y.append(y1)
            n += 1
            h_s.append(h)
            # continue
        else:
            cnt_rejected += 1

        h *= min(rho_max, max(rho_min, rho))        

    return h_s, y, t, cnt_accepted, cnt_rejected

# Adaptive Runge-Kutta method with embedded RK
def embedded_adaptive_RK(f, y_0, t_0, t_f, h_0, TOL, rho_min, rho_max, m, s, A, b1, b2, c):
    myMethod = rk.ExplicitRungeKuttaMethod(A, b1, c, name = "MyMethod")
    myEmbeddedMethod = rk.ExplicitRungeKuttaMethod(A, b2, c, name = "MyEmbeddedMethos")
    p = myMethod.order()
    pe = myEmbeddedMethod.order()

    cnt_accepted = 0
    cnt_rejected = 0

    h = h_0
    h_s = [h]
    f_ = lambda t, y: np.asarray(f(t, y))
    t = [t_0]
    y = [y_0]

    n = 0
    print(h)
    while t[n] < t_f:

        y1, y2, error = EE.embedded_RK(f_, h, y[n], t[n], s, A, b1, b2, c)
        rho = rho_max if error == 0 else m * (TOL / error) ** (1 / (p + 1))
        if error <= TOL:
            if t[n] + h == t[n]:
                warnings.warn("Step size became too small; stopping to avoid infinite loop.")
                break
            cnt_accepted += 1
            t.append(t[n] + h) 
            y.append(y1)
            n += 1
            h_s.append(h)
        else:
            cnt_rejected += 1

        h *= min(rho_max, max(rho_min, rho))        

    # print(y)
    return h_s, y, t, cnt_accepted, cnt_rejected

# adaptive Runge-Kutta method with using controllers
def adaptive_RK_with_controller(f, y_0, t_0, t_f, h_0, TOL, controller, s, A, b1, b2, c):
    myMethod = rk.ExplicitRungeKuttaMethod(A, b1, c, name = "MyMethod")
    myEmbeddedMethod = rk.ExplicitRungeKuttaMethod(A, b2, c, name = "MyEmbeddedMethos")
    p = myMethod.order()
    pe = myEmbeddedMethod.order()

    cnt_accepted = 0
    cnt_rejected = 0

    h = h_0
    h_s = [h]
    f_ = lambda t, y: np.asarray(f(t, y))
    t = [t_0]
    y = [y_0]
    
    error_prev = error_prev2 = TOL
    rho_prev = 1.0

    n = 0
    while t[n] < t_f:
        y1, y2, error = EE.embedded_RK(f_, h, y[n], t[n], s, A, b1, b2, c)

        rho = 0
        if controller == "I":
            rho = CS.I_controller(error, TOL, p)
        elif controller == "PI":
            rho = CS.PI_controller(error, error_prev, TOL, p) 
        elif controller == "PID":
            rho = CS.PID_controller(error, error_prev, error_prev2, TOL, p)
        elif controller == "PI3333":
            rho = CS.controller_PI3333(error, error_prev, TOL, p)
        elif controller == "H211":
            rho = CS.controller_H211b(error, error_prev, rho_prev, TOL, p)
        else:
            raise ValueError("Unknown controller")

        rho = max(0.1, min(1.5, rho))
       
        if error < TOL:
            if t[n] + h == t[n]:
                warnings.warn("Step size became too small; stopping to avoid infinite loop.")
                break
            cnt_accepted += 1
            t.append(t[n] + h)
            y.append(y1)
            n += 1
            error_prev2 = error_prev
            error_prev = error
            rho_prev = rho
            h_s.append(h)
        else:
            cnt_rejected += 1

        # if abs(rho - 1.0) < 1e-10:
        #     warnings.warn("Step size change is too small; stopping to avoid infinite loop.")
        #     break

        h *= rho    

    return h_s, y, t, cnt_accepted, cnt_rejected