# __author: Bao Li__ #
import numpy as np
from math import sin, cos, pi


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