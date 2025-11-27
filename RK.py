import numpy as np

# A full RK method with given Butcher tableau
# Parameters:
# f - the differential equation
# y_0 - initial value
# t_0, t_f - the interval
# N - number of steps
# s - the number of stages
# A, b, c - the Butcher tableau

def RK(f, y_0, t_0, t_f, N, s, A, b, c):
    h = (t_f - t_0) / N
    t = np.linspace(t_0, t_f, N+1)
    y = np.zeros(N+1)
    y[0] = y_0
    
    k = np.zeros(s)

    for n in range(0,N):
        k[0] = f(t[n], y[n])
        for j in range(1, s):
            k[j] = f(t[n] + h * c[j], y[n] + h * np.dot(A[j], k))
        y[n+1] = y[n] + h * np.dot(b, k)

    return h, y, t

# One step of an RK method with fixed step size
# Parameters:
# f - the differential equation
# h - step size
# y_n - the current value
# t_n - the current step
# s - the number of stages
# A, b, c - the Butcher tableau

def RK_step(f, h, y_n, t_n, s, A, b, c):
    k = np.zeros(s)

    k[0] = f(t_n, y_n)
    for j in range(1, s):
        k[j] = f(t_n + h * c[j], y_n + h * np.dot(A[j], k))
    
    return y_n + h * np.dot(b, k)

def get_k_values(f, h, y_n, t_n, s, A, c):
    k = np.zeros(s)

    k[0] = f(t_n, y_n)
    for j in range(1, s):
        k[j] = f(t_n + h * c[j], y_n + h * np.dot(A[j], k))
    
    return k