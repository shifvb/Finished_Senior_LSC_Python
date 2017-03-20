import math
import numpy as np
import time

# PI = math.pi


PI = 3.1415926


def Initialize(L: np.ndarray, a: np.ndarray, b: np.ndarray, nRows: int, nCols: int, StepX: int, StepY: int,
               Color: float, Distance: float):
    print("\t[{}] [Initialize.py] step_1/3".format(time.ctime()[11:19]))
    vcos = np.vectorize(math.cos)
    vsin = np.vectorize(math.sin)
    thetaL = (np.resize(L.copy(), [nRows, nCols]) / 255.0 * PI / 2.0).astype(np.float64)
    thetaa = (np.resize(a.copy(), [nRows, nCols]) / 255.0 * PI / 2.0).astype(np.float64)
    thetab = (np.resize(b.copy(), [nRows, nCols]) / 255.0 * PI / 2.0).astype(np.float64)
    thetax = np.empty([nRows, nCols], dtype=np.float64)
    thetay = np.empty([nRows, nCols], dtype=np.float64)
    for i in range(thetax.shape[0]):
        thetax[i, :] = i
    for j in range(thetay.shape[1]):
        thetay[:, j] = j
    thetax = (thetax / StepX) * PI / 2
    thetay = (thetay / StepY) * PI / 2
    L1 = Color * vcos(thetaL)
    L2 = Color * vsin(thetaL)
    a1 = Color * vcos(thetaa) * 2.55
    a2 = Color * vsin(thetaa) * 2.55
    b1 = Color * vcos(thetab) * 2.55
    b2 = Color * vsin(thetab) * 2.55
    x1 = Distance * vcos(thetax)
    x2 = Distance * vsin(thetax)
    y1 = Distance * vcos(thetay)
    y2 = Distance * vsin(thetay)

    print("\t[{}] [Initialize.py] step_2/3".format(time.ctime()[11:19]))
    size = nRows * nCols
    sigmaL1 = L1.sum() / size
    sigmaL2 = L2.sum() / size
    sigmaa1 = a1.sum() / size
    sigmaa2 = a2.sum() / size
    sigmab1 = b1.sum() / size
    sigmab2 = b2.sum() / size
    sigmax1 = x1.sum() / size
    sigmax2 = x2.sum() / size
    sigmay1 = y1.sum() / size
    sigmay2 = y2.sum() / size


    print("\t[{}] [Initialize.py] step_3/3".format(time.ctime()[11:19]))
    W = L1 * sigmaL1 + L2 * sigmaL2 + a1 * sigmaa1 + a2 * sigmaa2 + b1 * sigmab1 + \
        b2 * sigmab2 + x1 * sigmax1 + x2 * sigmax2 + y1 * sigmay1 + y2 * sigmay2
    L1 /= W
    L2 /= W
    a1 /= W
    a2 /= W
    b1 /= W
    b2 /= W
    x1 /= W
    x2 /= W
    y1 /= W
    y2 /= W
    return L1.astype(np.float32), L2.astype(np.float32), a1.astype(np.float32), \
           a2.astype(np.float32), b1.astype(np.float32), b2.astype(np.float32), \
           x1.astype(np.float32), x2.astype(np.float32), y1.astype(np.float32), \
           y2.astype(np.float32), W.astype(np.float64)
