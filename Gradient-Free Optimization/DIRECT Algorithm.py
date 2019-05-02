# __author: Bao Li__ #
import matplotlib.pyplot as plt
from sympy import *
from math import sqrt
import numpy as np
from scipy.spatial import ConvexHull


f_hist = []
f_min_hist = []
x_min_hist = []
d_hist = []
loc_hist = []
shape_hist = []


def rosenbrock(x):
    global FUNCCALLS
    FUNCCALLS += 1
    funcValue = 0
    gradient = []
    n = len(x)
    if n == 2:
        funcValue = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

        gradient = np.array([100 * 2 * (x[1] - x[0] ** 2) * (-2 * x[0]) - 2 * (1 - x[0]), 100 * 2 * (x[1] - x[0] ** 2)])
    else:
        for i in range(0, n - 1):
            f_i = 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
            funcValue = funcValue + f_i

            a = 100 * 2 * (x[i + 1] - x[i] ** 2) * (-2 * x[i]) - 2 * (1 - x[i])
            b = 100 * 2 * (x[i] - x[i - 1] ** 2)
            if i == 0:
                g = a
                gradient.append(g)
            elif 0 < i < (n - 2):
                g = a + b
                gradient.append(g)
            else:
                g = a + b
                gradient.append(g)
                g = 100 * 2 * (x[i + 1] - x[i] ** 2)
                gradient.append(g)
                gradient = np.array(gradient)
    return funcValue, gradient


def hullNew(points):
    hull = ConvexHull(points)
    a = points[hull.vertices, 0]
    b = points[hull.vertices, 1]
    hull_new = np.zeros([len(a), 2])

    for j in range(2):
        for i in range(len(a)):
            if j == 0:
                hull_new[i, j] = a[i]
            else:
                hull_new[i, j] = b[i]

    def findSmallest(arr,hull_new_1):
        smallest = arr[0]
        smallest_index = 0
        for i in range(1, len(arr)):
            if arr[i] > smallest:
                smallest = arr[i]
                smallest_index = i
            elif arr[i] == smallest:
                if hull_new_1[smallest_index] > hull_new_1[i]:
                    smallest_index = i
        return smallest_index

    def selectionSort(arr):
        global inddd
        newArr = []

        hull_new_0 = arr[:,0]
        hull_new_1 = arr[:,1]
        for i in range(len(hull_new_0)):
            smallest = findSmallest(hull_new_0, hull_new_1)
            v = arr[smallest]
            newArr.append(v)
            arr = np.delete(arr, smallest, axis=0)
            hull_new_0 = arr[:, 0]
            if i == 0:
                inddd = smallest
        return newArr, inddd

    def findSmallest1(arr):
        smallest = arr[0]
        smallest_index = 0
        for i in range(1, len(arr)):
            if arr[i] < smallest:
                smallest = arr[i]
                smallest_index = i
        return smallest_index

    def selectionSort1(arr):
        global inddd
        newArr = []

        a = arr[:, 0]
        b = arr[:, 1]
        for i in range(len(a)):
            smallest = findSmallest1(b)
            v = arr[smallest]
            newArr.append(v)
            arr = np.delete(arr, smallest, axis=0)
            b = arr[:, 1]
            if i == 0:
                inddd = smallest

        return newArr, inddd


    w = hull_new
    a_r, ind_r = selectionSort(w)
    a_l, ind_l = selectionSort1(w)

    a_low = hull_new[ind_l]
    b_high = hull_new[ind_r]
    k = (b_high[1]-a_low[1])/(b_high[0]-a_low[0])
    hull_n = []
    for i in range(len(hull_new)):
        a = hull_new[ind_l]
        b = hull_new[i]
        if b[0] != a[0]:
            k_test = (b[1]-a[1])/(b[0]-a[0])
            if k >= k_test:
                hull_n.append(b)

    hull_new = hull_n
    return hull_new


def direct(func, lowerBound, upperBound):
    """
    :param dim: dimension of x: i
    :param l: length of the hypoercubes or hyperrectrangles: n x dim matrix (n number of hyper)
    :param iter_max: number of iteration: k
    :param t: counter for how many times trisect on variable i over course of search: 1 x dim matrix
    :param f_min: optimum result
    :param x_min: optimum result
    :param i_min: direction to trisect and sample
    :param epsilon: The parameter balances the search between local and global search
    :param S: set of rectangles for refinement
    :param I: set of direction to trisect
    :param loc: set of locaton for hypoercubes or hyperrectrangles
    :param shapee: set of shapee for hypoercubes or hyperrectrangles
    """

    # Initialize parameters
    global k, t, f, f_min, i_min, x_min, f_min_r, f_min_l, \
        loc_min_l, loc_min_r, f_median, I, loc, d, delta
    dim = len(lowerBound)
    l = upperBound - lowerBound   # 1 x dim matrix
    iter_max = 400
    c_0 = 0.5*(upperBound + lowerBound)
    t = np.zeros(dim)
    delta = l/3
    e = np.zeros(dim)
    shapee = np.zeros(dim)

    # Initialize f_min, x_min, and i_min
    global  loc_1, loc_2

    for i in range(dim):
        e[i] = 1
        f_1 = func(c_0 + delta*e)[0]
        f_2 = func(c_0 - delta*e)[0]
        loc_1 = 0.5*np.ones(dim) + 1/3*e
        loc_2 = 0.5*np.ones(dim) - 1/3*e
        if f_1 > f_2:
            w = f_2
            w_x = c_0 - delta*e
        else:
            w = f_1
            w_x = c_0 + delta*e

        if i == 0:
            f_min_r = f_1
            f_min_l = f_2
            loc_min_r = loc_1
            loc_min_l = loc_2
            loc_1 = 0.5 * np.ones(dim)
            loc_2 = 0.5 * np.ones(dim)
            e = np.zeros(dim)

            f = func(c_0)[0]
            f_hist.append(f)
            loc = 0.5*np.ones(dim)
            loc_hist.append(loc)
            f_min = f
            f_min_hist.append(f_min)
            i_min = i
            x_min = c_0
            x_min_hist.append(x_min)

            d = sqrt(0.5**2*(dim - 1) + (1/3*0.5)**2)
            d_hist.append(d)
        elif i != 0 and w < f_min:
            f_min = w
            i_min = i
            f_min_r = f_1
            f_min_l = f_2
            x_min = w_x
            loc_min_r = loc_1
            loc_min_l = loc_2

    f_hist.append(f_min_l)
    d_r = d
    d_hist.append(d_r)
    f_hist.append(f_min_r)
    d_l = d
    d_hist.append(d_l)

    f_min_hist.append(f_min)
    x_min_hist.append(x_min)
    t[i_min] += 1  # count times trisect on variable i

    loc_hist.append(loc_min_l)
    loc_hist.append(loc_min_r)

    for i in range(dim):
        if t[i] == 0:
            shapee[i] = 1
        else:
            a = t[i]
            shapee[i] = 1/(3*a)

    shape_hist.append(shapee.copy())
    shape_hist.append(shapee.copy())
    shape_hist.append(shapee.copy())

    # Main iteration loop
    k = 0
    error = 1
    while error > 1e-6 and k < iter_max:

        if k == 0:
            # Set epsilon to small value
            f_median = np.median(f_hist)  # Calculate median for f
            epsilon = 1e-4*(f_median - f_min)
            f_star = f_min - epsilon
            f_hist.append(f_star)
            d_star = 0
            d_hist.append(d_star)
            shapee = np.zeros(dim)
            shape_hist.append(shapee.copy())
            loc = np.zeros(dim)
            loc_hist.append(loc)
        elif k!=0:
            # Set epsilon to small value
            f_median = np.median(f_hist)  # Calculate median for f
            epsilon = 1e-4*(f_median - f_min)
            f_star = f_min - epsilon
            f_hist[3] = f_star

        # Use “modified convex hull selection method”
        # to select a set 4 of rectangles for refinement.
        # Crest point
        pts = np.zeros([len(f_hist), 2]) # x = d_hist , y = f_hist
        for i in range(len(f_hist)):
            pts[i, 0] = d_hist[i]
            pts[i, 1] = f_hist[i]

        S = hullNew(pts)
        num_s = []
        shape_s = []
        f_s = []
        loc_s = []
        d_s = []
        for i in range(len(S)):
            s = S[i]
            for j in range(len(f_hist)):
                if s[1] == f_hist[j]:
                    num_s.append(j)
                    shape_s.append(shape_hist[j].copy())
                    f_s.append(f_hist[j])
                    loc_s.append(loc_hist[j])
                    d_s.append(d_hist[j])

        # Select a long side for trisection
        l_max = 0
        l_max_t = 0
        l_max_s = []
        for j in range(len(num_s)):
            shapee = shape_s[j]
            for i in range(len(shapee)):
                if i == 0:
                    l_max = shapee[i]
                    l_max_t = t[i]
                    i_min = i
                elif l_max < shapee[i]:
                    l_max = shapee[i]
                    l_max_t = t[i]
                    i_min = i
                elif l_max == shapee[i]:
                    l_t = t[i]
                    if l_max_t > l_t:
                        l_max = shapee[i]
                        i_min = i
                        l_max_t = l_t
            l_max_s.append(i_min)
            t[i_min] += 1

        # Trisect on that side and increment the appropriate t value
        for j in range(len(num_s)):
            loc = loc_s[j]
            f = f_s[j]
            dirc = l_max_s[j]
            d = d_s[j]
            x = np.zeros(dim)
            e = np.zeros(dim)
            for i in range(dim):
                x[i] = loc[i]*l[i] + lowerBound[i] # find x

            sh = shape_s[j]
            delta = (sh[dirc]/3)*l[dirc]
            e[dirc] = 1
            x_1 = x + delta*e
            x_2 = x - delta * e

            # update f_hist, loc_hist, d_hist, shape_hist
            # f_min_hist, x_min_hist, x_min, f_min
            loc_1 = loc + (sh[dirc]/3)*e
            loc_hist.append(loc_1)
            loc_2 = loc - (sh[dirc]/3)*e
            loc_hist.append(loc_2)

            f_1 = func(x_1)[0]
            f_hist.append(f_1)
            f_2 = func(x_2)[0]
            f_hist.append(f_2)

            a = sh[dirc]/3
            sh[dirc] = a
            a = num_s[j]
            shape_hist[a] = sh
            shape_hist.append(sh)
            shape_hist.append(sh)

            w = 0
            for i in range(dim):
                w += sh[i]**2
            d = sqrt(w)
            d_hist[a] = d
            d_hist.append(d)
            d_hist.append(d)

            if f != f_1 and f != f_2:
                if f == min(f, f_1, f_2) and f_min > f:
                    f_min = f
                    f_min_hist.append(f_min)
                    x_min = x
                    x_min_hist.append(x_min)
                elif f_1 == min(f, f_1, f_2) and f_min > f_1:
                    f_min = f_1
                    f_min_hist.append(f_min)
                    x_min = x_1
                    x_min_hist.append(x_min)
                elif f_2 == min(f, f_1, f_2) and f_min > f_2:
                    f_min = f_2
                    f_min_hist.append(f_min)
                    x_min = x_2
                    x_min_hist.append(x_min)
            elif f == f_1 and f != f_2:
                if f < f_2 and f_min > f:
                    f_min = f
                    f_min_hist.append(f_min)
                    x_min = x
                    x_min_hist.append(x_min)
                elif f > f_2 and f_min > f_2:
                    f_min = f_2
                    f_min_hist.append(f_min)
                    x_min = x_2
                    x_min_hist.append(x_min)
            elif f == f_2 and f != f_1:
                if f < f_1 and f_min > f:
                    f_min = f
                    f_min_hist.append(f_min)
                    x_min = x
                    x_min_hist.append(x_min)
                elif f > f_1 and f_min > f_1:
                    f_min = f_1
                    f_min_hist.append(f_min)
                    x_min = x_1
                    x_min_hist.append(x_min)
            elif f_1 == f_2 and f != f_1:
                if f < f_1 and f_min > f:
                    f_min = f
                    f_min_hist.append(f_min)
                    x_min = x
                    x_min_hist.append(x_min)
                elif f > f_1 and f_min > f_1:
                    f_min = f_1
                    f_min_hist.append(f_min)
                    x_min = x_1
                    x_min_hist.append(x_min)
            elif f_1 == f_2 == f and f_min > f:
                f_min = f_1
                f_min_hist.append(f_min)
                x_min = x_1
                x_min_hist.append(x_min)
        error = abs(f_min_hist[-2] - f_min_hist[-1])
        if error == 0:
            error = 1
        k += 1
        print("the "+str(k)+" iterations："+str(f_min))

    xopt = x_min_hist[-1]
    fopt = f_min_hist[-1]
    output = {'alias': 'Bao_Ch'}

    return xopt, fopt, output


def main():
    n = 2
    func = rosenbrock
    xopt, fopt, output = direct(func, lowerBound=np.ones(n)*-6, upperBound=np.ones(n)*5)
    print(xopt, fopt, output)
    print(FUNCCALLS)
    return xopt, fopt, output


if __name__ == "__main__":
    FUNCCALLS = 0
    main()
