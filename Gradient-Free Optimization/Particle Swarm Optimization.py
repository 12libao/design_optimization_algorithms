# __author: Bao Li__ #
from sympy import *
import numpy as np
from numpy import random


x_hist = []
f_hist = []


def gradFreeOpt(func, lowerBound, upperBound):
    """
    :param dim: dimension of x n
    :param iter_max:  k
    :param particle_num: i
    :param alpha:
    :param w:
    :param c1:
    :param c2:
    :param func:

    """
    alpha = 0.8
    iter_max = 100
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
            f_best = func(x_best)
        else:
            f_0 = func(x0[i])
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
            v_k_plus[i] = w*v_k[i] + c1*random.random()*(x_best_i[i] - x_k[i]) + c2*random.random()*(x_best - x_k[i])
            x_k_plus[i] = x_k[i] + alpha*v_k_plus[i]

            # Update the particle position while enforcing bounds.
            n = 0
            while n < dim:
                if x_k_plus[i, n] < lowerBound[n] or x_k_plus[i, n] > upperBound[n]:
                    x_k_plus[i] = x_k[i] - alpha * v_k_plus[i]
                    v_k_plus[i] = c1 * random.random() * (x_best_i[i] - x_k[i]) + c2 * random.random() * (x_best - x_k[i])
                    x_k_plus[i] = x_k[i] + alpha * v_k_plus[i]
                    break
                n += 1

            # Update the best result
            f_k_plus = func(x_k_plus[i])
            f_best_i = func(x_best_i[i])
            if f_k_plus < f_best_i:
                x_best_i[i] = x_k_plus[i]
            if f_k_plus < f_best:
                x_best = x_k_plus[i]
                f_best = func(x_best)

            v_k[i] = v_k_plus[i]
            x_k[i] = x_k_plus[i]

        x_hist.append(x_best)
        f_hist.append(f_best)
        k += 1
        if len(f_hist) > 1:
            if abs(f_hist[-1] - f_hist[-2]) != 0:
                error = abs(f_hist[-1] - f_hist[-2])
                # print(error)

    xopt = x_hist[-1]
    fopt = f_hist[-1]
    output = {'alias': 'Bao_Ch'}
    return xopt, fopt, output