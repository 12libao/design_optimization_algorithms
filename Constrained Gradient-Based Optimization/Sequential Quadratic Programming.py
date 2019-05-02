# __author: Bao Li__ #
from sympy import *
from math import pi, sqrt
import numpy as np
from scipy.optimize import minimize
from optPostProcess import plotHist, plotContour


x_hist = []
f_hist = []
c_ineq_hist = []


def obj(x):
    x_hist.append(x.copy())
    A = x[0]
    S = x[1]
    # b = span, c = chord
    b = (A * S)**0.5
    c = S / b
    # Flow conditions and other xed variables
    rho = 1.23
    mu = 17.8e-6
    V = 35
    S_wet = 2.05 * S
    k = 1.2
    e = 0.96
    W_0 = 4940
    N_ult = 2.5
    t_c = 0.12

    Re = rho * V * c / mu
    C_f = 0.074 / Re ** 0.2

    error = 1
    W_w = 500
    while error > 1e-11:
        W_w_0 = W_w
        W = W_w_0 + W_0
        W_w = 45.42 * S + (8.71e-5 * N_ult * b ** 3) * (W_0 * W)**0.5 / (S * t_c)
        error = abs(W_w - W_w_0)

    W = W_w + W_0
    C_L = W / (0.5 * rho * V ** 2 * S)
    C_D = 0.03062702 / S + k * C_f * S_wet / S + C_L ** 2 / (pi * A * e)

    f_hist.append(C_D)
    C_D = 0.5*rho*V**2*S*C_D
    #print('C_D', C_D)
    #a = 1e-10
    #C_D = a*C_D
    return C_D


def jac_obj(x):

    def FD_central(x):
        n_in = len(x)
        J = np.zeros(2)
        h = 1e-4

        for i in range(n_in):
            dota_x = max(h, h * x[i])
            x[i] += dota_x
            f_plus = obj(x)
            x[i] -= 2 * dota_x
            f_minus = obj(x)
            J[i] = (f_plus - f_minus) / (2 * dota_x)
            x[i] += dota_x

        return J

    dfdx = FD_central(x)
    return dfdx


def con_ineq(x):
    A = x[0]
    S = x[1]
    b = (A * S)**0.5

    # Flow conditions and other xed variables
    rho = 1.23
    W_0 = 4940
    N_ult = 2.5
    t_c = 0.12
    C_L_max = 2.0
    V_min_required = 22

    error = 1
    W_w = 500
    while error > 1e-11:
        W_w_0 = W_w
        W = W_w_0 + W_0
        W_w = 45.42 * S + (8.71e-5 * N_ult * b ** 3) * sqrt(W_0 * W) / (S * t_c)
        error = abs(W_w - W_w_0)
    W = W_w + W_0

    V_min = (2 * W / (rho * S * C_L_max))**0.5
    print('V-min', V_min)
    c = np.zeros(1)
    c[0] = V_min_required - V_min
    c_ineq_hist.append(c)
    #print('c', c)
    return c


def jac_con_ineq(x):

    def FD_central(x):
        n_in = len(x)
        J = np.zeros(2)
        h = 1e-4

        for i in range(n_in):
            dota_x = max(h, h * x[i])
            x[i] += dota_x
            f_plus = con_ineq(x)
            x[i] -= 2 * dota_x
            f_minus = con_ineq(x)
            J[i] = (f_plus - f_minus) / (2 * dota_x)
            x[i] += dota_x

        return J

    dcdx = FD_central(x)
    return dcdx


# Logarithmic barrier method
def fun(x):
    a = obj(x)
    b = con_ineq(x)

    F = a - mu*np.log(b)
    # print(b)
    return F


def gf(x):

    def FD_central(x):
        n_in = len(x)
        J = np.zeros(2)
        h = 1e-4

        for i in range(n_in):
            dota_x = max(h, h * x[i])
            x[i] += dota_x
            f_plus = fun(x)
            x[i] -= 2 * dota_x
            f_minus = fun(x)
            J[i] = (f_plus - f_minus) / (2 * dota_x)
            x[i] += dota_x

        return J

    dfdx = FD_central(x)
    return dfdx


def mini(x0, ob, jac_ob, epsilon_g):
    global result
    result = []
    global result_x
    result_x = [x0]
    maxk = 1e5
    rho = 0.55
    sigma = 0.4
    gama = 0.7
    global k
    k = 0
    n = np.shape(x0)[0]
    Bk = np.eye(n)
    while k < maxk:
        gk = jac_ob(x0)
        if np.linalg.norm(gk) < epsilon_g:
            break
        dk = -1.0*np.linalg.solve(Bk,gk)
        m = 0
        mk = 0
        while m < 20:
            gk1 = jac_ob(x0 + rho**m*dk)
            if ob(x0+rho**m*dk) < ob(x0)+sigma*rho**m*np.dot(gk,dk) and np.dot(gk1.T, dk) >=  gama*np.dot(gk.T,dk):
                mk = m
                break
            m += 1
        x = x0 + rho**mk*dk
        s = con_ineq(x)
        if s < 0:
            break
        print ("the"+str(k)+"iterations："+str(x))
        sk = x - x0
        yk = jac_ob(x) - gk
        if np.dot(sk,yk) > 0:
            Bs = np.dot(Bk,sk)
            ys = np.dot(yk,sk)
            sBs = np.dot(np.dot(sk,Bk),sk)
            Bk = Bk - 1.0*Bs.reshape((n,1))*Bs/sBs + 1.0*yk.reshape((n,1))*yk/ys
        k += 1

        #dota_f = np.log(abs(ob(x0, mu) - ob(x, mu)))
        x0 = x
        #result.append(dota_f)
        result_x.append(x0)
        global xopt
        global fopt
        xopt = result_x[-1]
        fopt = ob(xopt)
    return xopt, fopt

def lb(x):
    rho = 0.5
    error = 1
    f = fun(x)
    while error > 1e-8:
        a = x
        b = mu
        x_plus, f_plus = mini(x0 = a, ob = fun, jac_ob = gf, epsilon_g = 1e-8)
        s = con_ineq(x_plus)
        if s < 0:
            break
        #print(x_plus, f_plus)
        mu = rho*b
        #print(mu)
        x = x_plus
        error = abs(f - f_plus)
        f = f_plus
    return x


q = [10.0, 20.0]
global mu
mu = 1


ineq_cons = {'type': 'ineq', 'jac': jac_con_ineq,
        'fun': con_ineq}
method = 'slsqp'
x0 = np.array([10.0, 20.0])
bnds = ((5,25), (5, 25))
res = minimize(obj, x0, jac = jac_obj, method=method, bounds = bnds, tol=1e-8,
                options={'ftol': 1e-8,'disp': True}, constraints = ineq_cons)

print(res)
plotContour(obj=fun,obj1=obj ,bnds=bnds, x_hist=x_hist, con_ineq=con_ineq)