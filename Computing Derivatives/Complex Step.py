# __author: Bao Li__ #
import numpy as np


def complex_step(f, x):
    x = np.array(x, dtype=complex)

    # dmass_dA : ndarray of length 10: derivative of mass w.r.t. each A
    f_0 = np.append(f(x)[0], f(x)[1])
    n_in = len(x)
    n_out = len(f_0)
    J = np.zeros((n_out, n_in), dtype=complex)
    h = 1e-30

    for i in range(n_out):
        for j in range(n_in):
            x[j] += complex(0, h)
            y = np.zeros((1, 11), dtype=complex)
            y[0, 0] = f(x)[0]
            y[0, 1:11] = f(x)[1]
            J[i, j] = y[0, i].imag / h
            x[j] -= complex(0, h)

    J_m = J[0, :]
    J_s = J[1:]

    return J_m, J_s