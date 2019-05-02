# __author: Bao Li__ #
import numpy as np

def FD_forward(f, x):
    # dmass_dA : ndarray of length 10: derivative of mass w.r.t. each A
    f_0 = np.append(f(x)[0], f(x)[1])
    n_in = len(x)
    n_out = len(f_0)
    J = np.zeros((n_out, n_in))
    h = 1e-6

    for i in range(n_out):
        for j in range(n_in):
            dota_x = max(h, h * x[j])
            x[j] += dota_x
            f_plus = np.append(f(x)[0], f(x)[1])
            J[i, j] = (f_plus[i] - f_0[i]) / dota_x
            x[j] -= dota_x

    J_m = J[0, :]
    J_s = J[1:]

    return J_m, J_s


def FD_central(f, x):
    # dmass_dA : ndarray of length 10: derivative of mass w.r.t. each A
    f_0 = np.append(f(x)[0], f(x)[1])
    n_in = len(x)
    n_out = len(f_0)
    J = np.zeros((n_out, n_in))
    h = 1e-4

    for i in range(n_out):
        for j in range(n_in):
            dota_x = max(h, h * x[j])
            x[j] += dota_x
            f_plus = np.append(f(x)[0], f(x)[1])
            x[j] -= 2 * dota_x
            f_minus = np.append(f(x)[0], f(x)[1])
            J[i, j] = (f_plus[i] - f_minus[i]) / (2 * dota_x)
            x[j] += dota_x

    J_m = J[0, :]
    J_s = J[1:]

    return J_m, J_s