# __author: Bao Li__ #
import numpy as np


def quadraticInterpolation(a, h, h0, g0):

    numerator = g0 * a ** 2
    denominator = 2 * (g0 * a + h0 - h)
    if abs(denominator) < 1e-12:
        return a
    return numerator / denominator


def LineSearch(fun, dfun, theta, d):
    a0 = 1
    c1 = 1e-4
    c2 = 0.9
    a_min = 1e-7
    max_iter = 1e5

    eps = 1e-16
    c1 = min(c1, 0.5)
    a_pre = 0
    a_cur = a0
    f_val = fun(theta)
    g_val = np.sum(dfun(theta) * d.T)
    h_pre = f_val
    k = 0
    while k < max_iter and abs(a_cur - a_pre) >= eps:
        h_cur = fun(theta + a_cur * d)
        if h_cur > f_val + c1 * a_cur * g_val or h_cur >= h_pre and k > 0:
            return zoom(fun, dfun, theta, d, a_pre, a_cur, c1, c2)
        g_cur = np.sum(dfun(theta + a_cur * d) * d.T)
        if abs(g_cur) <= -c2 * g_val:
            return a_cur
        if g_cur >= 0:
            return zoom(fun, dfun, theta, d, a_pre, a_cur, c1, c2)
        a_new = quadraticInterpolation(a_cur, h_cur, f_val, g_val)
        a_pre = a_cur
        a_cur = a_new
        h_pre = h_cur
        k += 1
    return a_min


def zoom(fun, dfun, theta, d, a_low, a_high, c1=1e-3, c2=0.9, max_iter=1e4):

    eps = 1e-16
    h = fun(theta)
    g = np.sum(dfun(theta) * d.T)
    k = 0
    h_low = fun(theta + a_low * d)
    h_high = fun(theta + a_high * d)
    while k < max_iter and abs(a_high - a_low) >= eps:
        a_new = (a_low + a_high) / 2
        h_new = fun(theta + a_new * d)
        if h_new > h + c1 * a_new * g or h_new > h_low:
            a_high = a_new
            h_high = h_new
        else:
            g_new = np.sum(dfun(theta + a_new * d) * d.T)
            if abs(g_new) <= -c2 * g:
                return a_new
            if g_new * (a_high - a_low) >= 0:
                a_high = a_new
                h_high = h_new
            else:
                a_low = a_new
                h_low = h_new
        k += 1
    return a_low