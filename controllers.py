import numpy as np

def I_controller(E, tol, p):
    if E == 0:
        return 2.0
    return 0.9 * (tol / E)**(1/(p+1))

def PI_controller(E, E_prev, tol, p):
    if E == 0:
        return 2.0
    kI = 0.7/(p+1)
    kP = 0.4/(p+1)
    return 0.9 * (tol/E)**kI * (E_prev/E)**kP

def PID_controller(E, E_prev, E_prev2, tol, p):
    if E == 0:
        return 2.0
    kI = 1/(p+1)
    kP = -0.5/(p+1)
    kD = -0.1/(p+1)
    return 0.9 * (tol/E)**kI * (E_prev/E)**kP * ((E_prev**2)/(E*E_prev2))**kD

def controller_PI3333(E, E_prev, tol, p):
    eps = 1e-30
    E = max(E, eps)
    E_prev = max(E_prev, eps)
    beta1, beta2, alpha = 2/3, -1/3, 0.0
    k = p + 1
    return (tol / E)**(beta1 / k) * (tol / E_prev)**(beta2 / k)

def controller_H211b(E, E_prev, rho_prev, tol, p, b=4.0):
    eps = 1e-30
    E = max(E, eps)
    E_prev = max(E_prev, eps)
    beta1 = 1.0 / b
    beta2 = 1.0 / b
    alpha = 1.0 / b
    k = p + 1
    return (tol / E)**(beta1 / k) * (tol / E_prev)**(beta2 / k) * (rho_prev**alpha)