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