# __author: Bao Li__ #
import numpy as np
from math import sin, cos, pi
from scipy.optimize import minimize


def bar(E, A, L, phi):
    """Computes the stiffness and stress matrix for one element

    Parameters
    ----------
    E : float
        modulus of elasticity
    A : float
        cross-sectional area
    L : float
        length of element
    phi : float
        orientation of element

    Outputs
    -------
    K : 4 x 4 ndarray
        stiffness matrix
    S : 1 x 4 ndarray
        stress matrix

    """

    # rename
    c = cos(phi)
    s = sin(phi)

    # stiffness matrix
    k0 = np.array([[c ** 2, c * s], [c * s, s ** 2]])
    k1 = np.hstack([k0, -k0])
    K = E * A / L * np.vstack([k1, -k1])

    # stress matrix
    S = E / L * np.array([[-c, -s, c, s]])

    return K, S


def node2idx(node, DOF):
    """Computes the appropriate indices in the global matrix for
    the corresponding node numbers.  You pass in the number of the node
    (either as a scalar or an array of locations), and the degrees of
    freedom per node and it returns the corresponding indices in
    the global matrices

    """

    idx = np.array([], dtype=np.int)

    for i in range(len(node)):
        n = node[i]
        start = DOF * (n - 1)
        finish = DOF * n

        idx = np.concatenate((idx, np.arange(start, finish, dtype=np.int)))

    return idx


def truss(A):
    global FUNCCALLS
    FUNCCALLS += 1
    """Computes mass and stress for an arbitrary truss structure

    Parameters
    ----------
    start : ndarray of length nbar
        index of start of bar (1-based indexing) start and finish can be in any order as long as consistent with phi
    finish : ndarray of length nbar
        index of other end of bar (1-based indexing)
    phi : ndarray of length nbar (radians)
        defines orientation or bar
    A : ndarray of length nbar
        cross-sectional areas of each bar
    L : ndarray of length nbar
        length of each bar
    E : ndarray of length nbar
        modulus of elasticity of each bar
    rho : ndarray of length nbar
        material density of each bar
    Fx : ndarray of length nnode
        force in the x-direction at each node
    Fy : ndarray of length nnode
        force in the y-direction at each node
    rigid : list(boolean) of length nnode
        True if node_i is rigidly constrained

    Outputs
    -------
    mass : float
        mass of the entire structure
    stress : ndarray of length nbar
        stress of each bar

    """
    start = np.array([5, 3, 6, 4, 4, 2, 5, 6, 3, 4])
    finish = np.array([3, 1, 4, 2, 3, 1, 4, 3, 2, 1])
    phi = np.array([0, 0, 0, 0, pi / 2, pi / 2, -pi / 4, pi / 4, -pi / 4, pi / 4])
    L = np.array([10, 10, 10, 10, 10, 10, 2 ** 0.5 * 10, 2 ** 0.5 * 10, 2 ** 0.5 * 10, 2 ** 0.5 * 10])
    E = np.ones(10) * 70 * 10 ** 9
    rho = np.ones(10) * 2720
    Fx = np.zeros(6)
    Fy = np.array([0, -5 * 10 ** 5, 0, -5 * 10 ** 5, 0, 0])
    rigid = [False, False, False, False, True, True]

    n = len(Fx)  # number of nodes
    DOF = 2  # number of degrees of freedom
    nbar = len(A)  # number of bars

    # mass
    mass = np.sum(rho * A * L)

    # stiffness and stress matrices
    if func == complex_step:
        K = np.zeros((DOF * n, DOF * n), dtype=complex)
        S = np.zeros((nbar, DOF * n), dtype=complex)
    else:
        K = np.zeros((DOF * n, DOF * n))
        S = np.zeros((nbar, DOF * n))

    for i in range(nbar):  # loop through each bar

        # compute submatrix for each element
        Ksub, Ssub = bar(E[i], A[i], L[i], phi[i])

        # insert submatrix into global matrix
        idx = node2idx([start[i], finish[i]], DOF)  # pass in the starting and ending node number for this element
        K[np.ix_(idx, idx)] += Ksub
        S[i, idx] = Ssub

    # applied loads
    F = np.zeros((n * DOF, 1))

    for i in range(n):
        idx = node2idx([i + 1], DOF)  # add 1 b.c. made indexing 1-based for convenience
        F[idx[0]] = Fx[i]
        F[idx[1]] = Fy[i]

    # boundary condition
    idx = np.squeeze(np.where(rigid))
    remove = node2idx(idx + 1, DOF)  # add 1 b.c. made indexing 1-based for convenience

    K = np.delete(K, remove, axis=0)
    K = np.delete(K, remove, axis=1)
    F = np.delete(F, remove, axis=0)
    S = np.delete(S, remove, axis=1)

    # solve for deflections
    d = np.linalg.solve(K, F)

    # compute stress
    stress = np.dot(S, d).reshape(nbar)

    return mass, stress


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


def adjoint(f, x):
    A = x
    start = np.array([5, 3, 6, 4, 4, 2, 5, 6, 3, 4])
    finish = np.array([3, 1, 4, 2, 3, 1, 4, 3, 2, 1])
    phi = np.array([0, 0, 0, 0, pi / 2, pi / 2, -pi / 4, pi / 4, -pi / 4, pi / 4])
    L = np.array([10, 10, 10, 10, 10, 10, 2 ** 0.5 * 10, 2 ** 0.5 * 10, 2 ** 0.5 * 10, 2 ** 0.5 * 10])
    E = np.ones(10) * 70 * 10 ** 9
    rho = np.ones(10) * 2720
    Fx = np.zeros(6)
    Fy = np.array([0, -5 * 10 ** 5, 0, -5 * 10 ** 5, 0, 0])
    rigid = [False, False, False, False, True, True]

    _, stress = f(A)
    n = len(Fx)  # number of nodes
    DOF = 2  # number of degrees of freedom
    nbar = len(A)  # number of bars

    # dmassdA
    dmassdA = np.array(rho*L)
    # stiffness and stress matrices
    K = np.zeros((DOF*n, DOF*n))
    S = np.zeros((nbar, DOF*n))

    for i in range(nbar):  # loop through each bar

        # compute submatrix for each element
        Ksub, Ssub = bar(E[i], A[i], L[i], phi[i])

        # insert submatrix into global matrix
        idx = node2idx([start[i], finish[i]], DOF)  # pass in the starting and ending node number for this element
        K[np.ix_(idx, idx)] += Ksub
        S[i, idx] = Ssub

    # applied loads
    F = np.zeros((n * DOF, 1))

    for i in range(n):
        idx = node2idx([i + 1], DOF)  # add 1 b.c. made indexing 1-based for convenience
        F[idx[0]] = Fx[i]
        F[idx[1]] = Fy[i]

    # boundary condition
    idx = np.squeeze(np.where(rigid))
    remove = node2idx(idx + 1, DOF)  # add 1 b.c. made indexing 1-based for convenience
    K = np.delete(K, remove, axis=0)
    K = np.delete(K, remove, axis=1)
    F = np.delete(F, remove, axis=0)
    S = np.delete(S, remove, axis=1)

    # solve for deflections
    d = np.linalg.solve(K, F)
    global  dstressdA
    dstressdA = np.zeros((10, 10))

    for i in range(nbar):  # loop through each bar

        # compute submatrix for each element
        dK_dA = np.zeros((DOF * n, DOF * n))
        Ksub, _ = bar(E[i], A[i], L[i], phi[i])
        Ksub = Ksub/A[i]

        # insert submatrix into global matrix
        idx = node2idx([start[i], finish[i]], DOF)  # pass in the starting and ending node number for this element
        dK_dA[np.ix_(idx, idx)] = Ksub
        dK_dA = np.delete(dK_dA, remove, axis=0)
        dK_dA = np.delete(dK_dA, remove, axis=1)

        K_1 = np.linalg.inv(K)
        dstress_dA = np.dot(-S, K_1)
        dstress_dA = np.dot(dstress_dA, dK_dA)
        dstress_dA = np.dot(dstress_dA, d)
        dstressdA[:, i] = dstress_dA[:, 0]

    return dmassdA, dstressdA


def tenbartruss(A, grad_type):
    """This is the subroutine for the 10-bar truss.  You will need to complete it.

    Parameters
    ----------
    A : ndarray of length 10
        cross-sectional areas of all the bars
    grad_type : string (optional)
        gradient type.  'FD' for finite difference, 'CS' for complex step,
        'AJ' for adjoint

    Outputs
    -------
    mass : float
        mass of the entire structure
    stress : ndarray of length 10
        stress of each bar
    dmass_dA : ndarray of length 10
        derivative of mass w.r.t. each A
    dstress_dA : 10 x 10 ndarray
        dstress_dA[i, j] is derivative of stress[i] w.r.t. A[j]

    """
    # --- setup 10 bar truss ----

    # we need setup (start, finish, phi, L, E, rho, Fx, Fy, rigid)
    # we have n = 6 nodes, m = 10 elements
    # n = len(Fx) = 6 # number of nodes
    # DOF = 2  # number of degrees of freedom
    # nbar = len(A) = 10  # number of bars

    # --- call truss function ----
    mass, stress = truss(A)
    # --- compute derivatives for provided grad_type ----
    J_m, J_s = grad_type(f=truss, x=A)
    dmass_dA = J_m
    dstress_dA = J_s

    if grad_type == complex_step:
        stress = np.array(stress.real, dtype=float)
        dmass_dA = np.array(dmass_dA.real, dtype=float)
        dstress_dA = np.array(dstress_dA.real, dtype=float)

    return mass, stress, dmass_dA, dstress_dA


# 3.1 test
grad_type = [FD_forward,FD_central, complex_step]


for _, func in enumerate(grad_type):
    FUNCCALLS = 0

    A = np.ones(10)*5*1e-4
    mass0, stress0, dmass_dA0, dstress_dA0 = tenbartruss(A, adjoint)
    mass, stress, dmass_dA, dstress_dA = tenbartruss(A, func)
    error = np.zeros((10))
    merror = 0
    for i in range(10):

        error[i] = (dmass_dA[i] - dmass_dA0[i])/dmass_dA0[i]
        merror += error[i]
    aerror = merror/10
    print(aerror)

for _, func in enumerate(grad_type):
    FUNCCALLS = 0

    A = np.ones(10) * 5 * 1e-2
    mass0, stress0, dmass_dA0, dstress_dA0 = tenbartruss(A, adjoint)
    mass, stress, dmass_dA, dstress_dA = tenbartruss(A, func)
    error = np.zeros((10,10))
    merror = 0
    for i in range(10):
        for j in range(10):
            error[i,j] = (dstress_dA[i,j] - dstress_dA0[i,j])/dstress_dA0[i,j]
            merror += error[i,j]

    aerror = merror/100
    print(aerror)


# 3.2 Truss Optimization
# Define objective function
x_hist = []
f_hist = []
c_ineq_hist = []


grad = adjoint

FUNCCALLS = 0

# a = 0.00000000001
a = 0.0000000001
def obj(x):
    """The objective function"""
    # Append current x value to our history list
    x_hist.append(x.copy())
    mass, _, _, _ = tenbartruss(x, grad)

    # Append current f value to our history list

    f_hist.append(mass)

    c = mass*a
    print(mass)
    return c


# Provide scipy the objective gradient (jacobian)
def jac_obj(x):
    """The gradient of the objective function"""
    _, _, dmass_dA, _ = tenbartruss(x, grad)
    c = dmass_dA*a
    return c

def con_ineq(x):

    c = np.zeros(10)
    _, stress, _, _ = tenbartruss(x, grad)

    for i in range(10):
        if i == 8:
            if stress[i] > 0:
                c[i] = 520*10**6 - stress[i]
            else:
                c[i] = stress[i] + 520*10**6
        else:
            if stress[i] > 0:
                c[i] = 170*10**6 - stress[i]
            else:
                c[i] = stress[i] + 170*10**6

    c_ineq_hist.append(c.copy())
    print(stress)
    return c


def jac_con_ineq(x):
    _, _, _, dstress_dA = tenbartruss(x, grad)
    return dstress_dA


ineq_cons = {'type': 'ineq', 'jac': jac_con_ineq,
             'fun': con_ineq}


# Some available Scipy.optimize algorithms
method = 'slsqp'
# Initial design variable
x0 = np.ones(10) * 2*10**-4
# Design variable bounds
#    ((x0_l, x0_u),  (x1_l, x1_u))
bnds = ((5e-5, 1), (5e-5, 1), (5e-5, 1), (5e-5, 1),
        (5e-5, 1), (5e-5, 1), (5e-5, 1), (5e-5, 1),
        (5e-5, 1), (5e-5, 1))

# Now we provide constraint information as well..
res = minimize(obj, x0, jac=jac_obj, method=method, bounds=bnds,
                 constraints=ineq_cons,options={'ftol':1e-8, 'disp':True})
print(res)
print(FUNCCALLS)
print(f_hist)
