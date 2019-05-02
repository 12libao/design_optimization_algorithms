# __author: Bao Li__ #

from math import pi
from scipy.optimize import minimize
from numpy import random
from sympy import *
from math import sqrt
import numpy as np


x_hist = []
f_hist = []

def obj(x):
    global FUNCCALLS

    FUNCCALLS += 1
    x_hist.append(x.copy())
    A = x[0]
    S = x[1]
    # b = span, c = chord

    b = (abs(A * S)) ** 0.5
    c = S / b
    # print(c)
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
    if c <= 0.85:
        C_f = 1.328 / Re ** 0.5
    else:
        C_f = 0.074 / Re ** 0.2

    error = 1
    W_w = 500
    while error > 1e-11:
        W_w_0 = W_w
        W = W_w_0 + W_0
        W_w = 45.42 * S + (8.71e-5 * N_ult * b ** 3) * (W_0 * W) ** 0.5 / (S * t_c)
        error = abs(W_w - W_w_0)

    W = W_w + W_0
    C_L = W / (0.5 * rho * V ** 2 * S)
    C_D = 0.03062702 / S + k * C_f * S_wet / S + C_L ** 2 / (pi * A * e)

    f_hist.append(C_D)
    C_D = 0.5 * rho * V ** 2 * S * C_D
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


def obj_fun(x):
    f1 = obj(x)
    f2 = jac_obj(x)
    return f1, f2


def quasiNewton(func, x0, epsilon_g):
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
        gk = func(x0)[1]
        if np.linalg.norm(gk) < epsilon_g:
            break
        dk = -1.0 * np.linalg.solve(Bk, gk)
        m = 0
        mk = 0
        while m < 20:
            gk1 = func(x0 + rho ** m * dk)[1]
            if func(x0 + rho ** m * dk)[0] < func(x0)[0] + sigma * rho ** m * np.dot(gk, dk) and np.dot(gk1.T,
                                                                                                        dk) >= gama * np.dot(
                    gk.T, dk):
                mk = m
                break
            m += 1
        x = x0 + rho ** mk * dk
        # print ("the"+str(k)+"iterations："+str(x))
        sk = x - x0
        yk = func(x)[1] - gk
        if np.dot(sk, yk) > 0:
            Bs = np.dot(Bk, sk)
            ys = np.dot(yk, sk)
            sBs = np.dot(np.dot(sk, Bk), sk)
            Bk = Bk - 1.0 * Bs.reshape((n, 1)) * Bs / sBs + 1.0 * yk.reshape((n, 1)) * yk / ys
        k += 1
        dota_f = np.log(abs(func(x0)[0] - func(x)[0]), dtype=float)
        x0 = x
        result.append(dota_f)
        result_x.append(x0)
        global xopt
        global fopt
        xopt = result_x[-1]
        fopt = func(xopt)[0]

    return xopt, fopt


def PSO(func, lowerBound, upperBound):
    alpha = 0.8
    iter_max = 400
    particle_num = 20
    w = 0.5
    c1 = 2.05
    c2 = 2.05
    dim = len(lowerBound)

    x0 = np.zeros([particle_num, dim])
    x_best_i = np.zeros([particle_num, dim])
    x_best = np.zeros([1, dim])

    # Initialize position
    for i in range(particle_num):
        for n in range(dim):
            x0[i, n] = random.uniform(lowerBound[n], upperBound[n], 1)
        x_best_i[i] = x0[i]
        if i == 0:
            x_best = x0[i]
            global f_best
            f_best = func(x_best)[0]
        else:
            f_0 = func(x0[i])[0]
            if f_0 < f_best:
                x_best = x0[i]
                f_best = f_0
    x_k = x0
    x_hist.append(x_best)
    f_hist.append(f_best)

    # Initialize velocity
    v_max = (np.array(list(upperBound)) - np.array(list(lowerBound))) * 0.15
    v_k = np.array([random.uniform(-v_max_value, v_max_value, particle_num) for v_max_value in v_max]).T

    # Main iteration loop
    global k
    k = 0
    error = 1
    x_k_plus = np.zeros([particle_num, dim])
    v_k_plus = np.zeros([particle_num, dim])
    while error > 1e-6 and k < iter_max:
        for i in range(particle_num):
            v_k_plus[i] = w * v_k[i] + c1 * random.random() * (x_best_i[i] - x_k[i]) + c2 * random.random() * (
                        x_best - x_k[i])
            x_k_plus[i] = x_k[i] + alpha * v_k_plus[i]

            # Update the particle position while enforcing bounds.
            n = 0
            while n < dim:
                if x_k_plus[i, n] < lowerBound[n] or x_k_plus[i, n] > upperBound[n]:
                    x_k_plus[i] = x_k[i] - alpha * v_k_plus[i]
                    v_k_plus[i] = c1 * random.random() * (x_best_i[i] - x_k[i]) + c2 * random.random() * (
                                x_best - x_k[i])
                    x_k_plus[i] = x_k[i] + alpha * v_k_plus[i]
                    break
                n += 1

            # Update the best result
            f_k_plus = func(x_k_plus[i])[0]
            f_best_i = func(x_best_i[i])[0]
            if f_k_plus < f_best_i:
                x_best_i[i] = x_k_plus[i]
            if f_k_plus < f_best:
                x_best = x_k_plus[i]
                f_best = func(x_best)[0]

            v_k[i] = v_k_plus[i]
            x_k[i] = x_k_plus[i]

        x_hist.append(x_best)
        f_hist.append(f_best)
        k += 1
        if len(f_hist) > 1:
            if abs(f_hist[-1] - f_hist[-2]) != 0:
                error = abs(f_hist[-1] - f_hist[-2])
                # print(error)

    # print(k)
    xopt = x_hist[-1]
    fopt = f_hist[-1]
    return xopt, fopt


def PSO_DIV(func, lowerBound, upperBound):
    alpha = 1
    iter_max = 1000
    particle_num = 20
    w = 0.5
    c1 = 2.05
    c2 = 2.05
    dim = len(lowerBound)

    x0 = np.zeros([particle_num, dim])
    x_best_i = np.zeros([particle_num, dim])
    x_best = np.zeros([1, dim])

    # Initialize position
    for i in range(particle_num):
        for n in range(dim):
            x0[i, n] = random.uniform(lowerBound[n], upperBound[n], 1)
        x_best_i[i] = x0[i]
        if i == 0:
            x_best = x0[i]
            global f_best
            f_best = func(x_best)[0]
        else:
            f_0 = func(x0[i])[0]
            if f_0 < f_best:
                x_best = x0[i]
                f_best = f_0
    x_k = x0
    x_hist.append(x_best)
    f_hist.append(f_best)

    # Initialize velocity
    v_max = (np.array(list(upperBound)) - np.array(list(lowerBound))) * 0.15
    v_k = np.array([random.uniform(-v_max_value, v_max_value, particle_num) for v_max_value in v_max]).T

    # Main iteration loop
    global k
    k = 0
    error = 1
    x_k_plus = np.zeros([particle_num, dim])
    v_k_plus = np.zeros([particle_num, dim])
    while error > 1e-12 and k < iter_max:

        # Convergence factor "diversity"
        d_low = 5e-6
        d_high = 0.1
        L0 = np.zeros(dim)
        L = upperBound[0] - lowerBound[0]

        for n in range(dim):
            L0[n] = upperBound[n] - lowerBound[n]
            if L0[n] < L:
                L = L0[n]

        sum_i_n = np.zeros(particle_num)
        for i in range(particle_num):
            for n in range(dim):
                sum_n0 = (x_k[i, n] - sum(x_k[:, n]) / particle_num) ** 2
                sum_i_n[i] = sum_i_n[i] + sum_n0
            sum_i_n[i] = sqrt(sum_i_n[i])

        sum_sn = sum(sum_i_n[:])
        diversity = (1 / (particle_num * L)) * sum_sn
        # print(diversity)

        # Convergence factor "x"
        l = c1 + c2
        x = 2 / (abs(2 - l - sqrt(l ** 2 - 4 * l)))
        # print(x)

        # Diffusion and Attraction
        global dir
        if k == 0:
            dir = 1
        elif dir > 0 and diversity < d_low:
            dir = -1
        elif dir < 0 and diversity > d_high:
            dir = 1
        # print(dir)

        w_max = 0.9
        w_min = 0.4
        # x = w_max*(k*(w_max - w_min)/(iter_max))
        # x = (w_max - w_min)*(iter_max - k)/iter_max + w_min
        # Update the velocity and position
        for i in range(particle_num):
            v_k_plus[i] = x * (v_k[i] + dir * (
                        c1 * random.random() * (x_best_i[i] - x_k[i]) + c2 * random.random() * (x_best - x_k[i])))
            x_k_plus[i] = x_k[i] + alpha * v_k_plus[i]

            # Update the particle position while enforcing bounds.
            n = 0
            while n < dim:
                if x_k_plus[i, n] < lowerBound[n] or x_k_plus[i, n] > upperBound[n]:
                    x_k_plus[i] = x_k[i] - alpha * v_k_plus[i]
                    v_k_plus[i] = c1 * random.random() * (x_best_i[i] - x_k[i]) + c2 * random.random() * (
                                x_best - x_k[i])
                    x_k_plus[i] = x_k[i] + alpha * v_k_plus[i]
                    break
                n += 1

            # Update the best result
            f_k_plus = func(x_k_plus[i])[0]
            f_best_i = func(x_best_i[i])[0]
            if f_k_plus < f_best_i:
                x_best_i[i] = x_k_plus[i]
            if f_k_plus < f_best:
                x_best = x_k_plus[i]
                f_best = func(x_best)[0]
                # print(f_best)
                # print(k)
            v_k[i] = v_k_plus[i]
            x_k[i] = x_k_plus[i]

        x_hist.append(x_best)
        f_hist.append(f_best)
        k += 1
        if len(f_hist) > 1:
            if abs(f_hist[-1] - f_hist[-2]) != 0:
                error = abs(f_hist[-1] - f_hist[-2])
                # print(error)

    # print(k)
    xopt = x_hist[-1]
    fopt = f_hist[-1]
    return xopt, fopt


def main():
    # global lowerBound, upperBound
    func = obj_fun
    global FUNCCALLS

    lowerBound = np.ones(2) * 10
    upperBound = np.ones(2) * 20
    dim = 2
    x0 = np.zeros([1, dim])
    for n in range(dim):
        x0[0, n] = random.uniform(lowerBound[n], upperBound[n], 1)
    x0 = np.random.normal(size=2) * 3
    xopt, fopt = PSO_DIV(func, lowerBound, upperBound)
    print(xopt, fopt)
    print(FUNCCALLS)

    q = [10.0, 20.0]
    FUNCCALLS = 0
    xopt, fopt = quasiNewton(func, q, 1e-6)
    print(xopt, fopt)
    print(FUNCCALLS)

    FUNCCALLS = 0
    x0 = np.array([10.0, 20.0])
    bnds = ((5, 25), (5, 25))
    methods = ['slsqp', 'bfgs', 'Powell', 'Nelder-Mead']
    res = minimize(obj, x0, jac=jac_obj, method=methods[3], bounds=bnds, tol=1e-6,
                   options={'ftol': 1e-6, 'disp': True})
    print(res)
    print(FUNCCALLS)

    return xopt, fopt


if __name__ == "__main__":
    FUNCCALLS = 0
    main()