# __author: Bao Li__ #
import numpy as np


def uncon(func, x0, epsilon_g):
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
        dk = -1.0*np.linalg.solve(Bk,gk)
        m = 0
        mk = 0
        while m < 20:
            gk1 = func(x0 + rho**m*dk)[1]
            if func(x0+rho**m*dk)[0] < func(x0)[0]+sigma*rho**m*np.dot(gk,dk) and np.dot(gk1.T, dk) >=  gama*np.dot(gk.T,dk):
                mk = m
                break
            m += 1
        x = x0 + rho**mk*dk
        print ("the"+str(k)+"iterations："+str(x))
        sk = x - x0
        yk = func(x)[1] - gk
        if np.dot(sk,yk) > 0:
            Bs = np.dot(Bk,sk)
            ys = np.dot(yk,sk)
            sBs = np.dot(np.dot(sk,Bk),sk)
            Bk = Bk - 1.0*Bs.reshape((n,1))*Bs/sBs + 1.0*yk.reshape((n,1))*yk/ys
        k += 1
        dota_f = np.log(abs(func(x0)[0]-func(x)[0]), dtype=float)
        x0 = x
        result.append(dota_f)
        result_x.append(x0)
        global xopt
        global fopt
        xopt = result_x[-1]
        fopt = func(xopt)[0]
    output = {'alias': 'Bao Li'}
    return xopt, fopt, output