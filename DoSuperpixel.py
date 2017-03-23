from point import Point
from EnforceConnectivity import EnforceConnectivity
from preEnforceConnetivity import preEnforceConnectivity
import numpy as np
import sys
import time
from scipy.io import loadmat
from test_utils import compare_matrix
import math
from math import pow
import pickle

DBL_MAX = sys.float_info[0]  # max float value
TEST_INITIALIZATION = False
TEST_KMEANS_LABEL = False
FAKE_KMEANS_LABEL = False
TEST_PEC_LABEL = False
FAKE_EC_LABEL = False


# Perform weighted kmeans iteratively in the ten dimensional feature space.
def DoSuperpixel(L1: np.ndarray, L2: np.ndarray, a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray,
                 x1: np.ndarray, x2: np.ndarray, y1: np.ndarray, y2: np.ndarray,
                 W: np.ndarray, label: np.ndarray, seedArray: list, seedNum: int, nRows: int, nCols: int, StepX: int,
                 StepY: int, iterationNum: int, thresholdCoef: int, new_label: np.ndarray):
    if FAKE_KMEANS_LABEL:
        iterationNum = -1
    print("\t[{}] [DoSuperpixel.py]: Pre-treatment".format(time.ctime()[11:19]))
    dist = np.empty([nRows, nCols], dtype=np.float64)
    centerL1 = np.empty([seedNum], dtype=np.float64)
    centerL2 = np.empty([seedNum], dtype=np.float64)
    centera1 = np.empty([seedNum], dtype=np.float64)
    centera2 = np.empty([seedNum], dtype=np.float64)
    centerb1 = np.empty([seedNum], dtype=np.float64)
    centerb2 = np.empty([seedNum], dtype=np.float64)
    centerx1 = np.empty([seedNum], dtype=np.float64)
    centerx2 = np.empty([seedNum], dtype=np.float64)
    centery1 = np.empty([seedNum], dtype=np.float64)
    centery2 = np.empty([seedNum], dtype=np.float64)
    WSum = np.empty([seedNum], dtype=np.float64)
    clusterSize = np.empty([seedNum], dtype=np.int32)

    print("\t[{}] [DoSuperpixel.py]: Initialization".format(time.ctime()[11:19]))
    for i in range(seedNum):
        centerL1[i] = 0
        centerL2[i] = 0
        centera1[i] = 0
        centera2[i] = 0
        centerb1[i] = 0
        centerb2[i] = 0
        centerx1[i] = 0
        centerx2[i] = 0
        centery1[i] = 0
        centery2[i] = 0
        x = seedArray[i].x
        y = seedArray[i].y
        minX = int(0 if x - StepX // 4 <= 0 else x - StepX // 4)
        minY = int(0 if y - StepY // 4 <= 0 else y - StepY // 4)
        maxX = int(nRows - 1 if x + StepX // 4 >= nRows - 1 else x + StepX // 4)
        maxY = int(nCols - 1 if y + StepY // 4 >= nCols - 1 else y + StepY // 4)
        Count = 0
        for j in range(minX, maxX + 1):
            for k in range(minY, maxY + 1):
                Count += 1
                centerL1[i] += L1[j][k]
                centerL2[i] += L2[j][k]
                centera1[i] += a1[j][k]
                centera2[i] += a2[j][k]
                centerb1[i] += b1[j][k]
                centerb2[i] += b2[j][k]
                centerx1[i] += x1[j][k]
                centerx2[i] += x2[j][k]
                centery1[i] += y1[j][k]
                centery2[i] += y2[j][k]
        centerL1[i] /= Count
        centerL2[i] /= Count
        centera1[i] /= Count
        centera2[i] /= Count
        centerb1[i] /= Count
        centerb2[i] /= Count
        centerx1[i] /= Count
        centerx2[i] /= Count
        centery1[i] /= Count
        centery2[i] /= Count

    if TEST_INITIALIZATION:
        data = loadmat("test_27_DOS_Initialization_centers.mat")
        print(compare_matrix.compare_1D_array(centerL1, data["tCenterL1"].reshape([200]), 10, 1e-8), end="", flush=True)
        print(compare_matrix.compare_1D_array(centerL2, data["tCenterL2"].reshape([200]), 10, 1e-8), end="", flush=True)
        print(compare_matrix.compare_1D_array(centera1, data["tCentera1"].reshape([200]), 10, 1e-8), end="", flush=True)
        print(compare_matrix.compare_1D_array(centera2, data["tCentera2"].reshape([200]), 10, 1e-8), end="", flush=True)
        print(compare_matrix.compare_1D_array(centerb1, data["tCenterb1"].reshape([200]), 10, 1e-8), end="", flush=True)
        print(compare_matrix.compare_1D_array(centerb2, data["tCenterb2"].reshape([200]), 10, 1e-8), end="", flush=True)
        print(compare_matrix.compare_1D_array(centerx1, data["tCenterx1"].reshape([200]), 10, 1e-8), end="", flush=True)
        print(compare_matrix.compare_1D_array(centerx2, data["tCenterx2"].reshape([200]), 10, 1e-8), end="", flush=True)
        print(compare_matrix.compare_1D_array(centery1, data["tCentery1"].reshape([200]), 10, 1e-8), end="", flush=True)
        print(compare_matrix.compare_1D_array(centery2, data["tCentery2"].reshape([200]), 10, 1e-8), end="", flush=True)
    # exit()

    print("\t[{}] [DoSuperpixel.py]: K-means".format(time.ctime()[11:19]))
    for iteration in range(iterationNum + 1):
        print("\t\t[{}] [DoSuperpixel.py]: K-means_iter_{}_step_1".format(time.ctime()[11:19], iteration))
        for i in range(nRows):
            for j in range(nCols):
                dist[i][j] = DBL_MAX
        for i in range(seedNum):
            # print("\t\t[{}] [DoSuperpixel.py]: K-means_iter_{}_seed{}".format(time.ctime()[11:19], iteration, i))
            x = seedArray[i].x
            y = seedArray[i].y
            minX = int(0 if x - StepX <= 0 else x - StepX)
            minY = int(0 if y - StepY <= 0 else y - StepY)
            maxX = int(nRows - 1 if x + StepX >= nRows - 1 else x + StepX)
            maxY = int(nCols - 1 if y + StepY >= nCols - 1 else y + StepY)

            # my implementation start
            step1_min_x = minX
            step1_max_x = maxX + 1
            step1_min_y = minY
            step1_max_y = maxY + 1
            step1_vpow = np.vectorize(lambda _: _ * _)
            step1_L1_pow = step1_vpow(L1[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerL1[i])
            step1_L2_pow = step1_vpow(L2[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerL2[i])
            step1_a1_pow = step1_vpow(a1[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centera1[i])
            step1_a2_pow = step1_vpow(a2[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centera2[i])
            step1_b1_pow = step1_vpow(b1[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerb1[i])
            step1_b2_pow = step1_vpow(b2[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerb2[i])
            step1_x1_pow = step1_vpow(x1[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerx1[i])
            step1_x2_pow = step1_vpow(x2[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerx2[i])
            step1_y1_pow = step1_vpow(y1[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centery1[i])
            step1_y2_pow = step1_vpow(y2[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centery2[i])
            step1_D = step1_L1_pow + step1_L2_pow + step1_a1_pow + step1_a2_pow + step1_b1_pow + step1_b2_pow + \
                      step1_x1_pow + step1_x2_pow + step1_y1_pow + step1_y2_pow

            step1_if = (step1_D - dist[step1_min_x: step1_max_x, step1_min_y: step1_max_y] < 0).astype(np.uint16)
            step1_neg_if = 1 - step1_if
            new_label[step1_min_x: step1_max_x, step1_min_y: step1_max_y] *= step1_neg_if
            new_label[step1_min_x: step1_max_x, step1_min_y: step1_max_y] += (step1_if * i)

            dist[step1_min_x: step1_max_x, step1_min_y: step1_max_y] *= step1_neg_if
            step1_D_to_plus = step1_D * step1_if
            dist[step1_min_x: step1_max_x, step1_min_y: step1_max_y] += step1_D_to_plus
            # my implementation end

            # previous implementation start
            # for m in range(minX, maxX + 1):
            #     for n in range(minY, maxY + 1):
            #         D = (L1[m][n] - centerL1[i]) * (L1[m][n] - centerL1[i]) + \
            #             (L2[m][n] - centerL2[i]) * (L2[m][n] - centerL2[i]) + \
            #             (a1[m][n] - centera1[i]) * (a1[m][n] - centera1[i]) + \
            #             (a2[m][n] - centera2[i]) * (a2[m][n] - centera2[i]) + \
            #             (b1[m][n] - centerb1[i]) * (b1[m][n] - centerb1[i]) + \
            #             (b2[m][n] - centerb2[i]) * (b2[m][n] - centerb2[i]) + \
            #             (x1[m][n] - centerx1[i]) * (x1[m][n] - centerx1[i]) + \
            #             (x2[m][n] - centerx2[i]) * (x2[m][n] - centerx2[i]) + \
            #             (y1[m][n] - centery1[i]) * (y1[m][n] - centery1[i]) + \
            #             (y2[m][n] - centery2[i]) * (y2[m][n] - centery2[i])
            #         if D < dist[m][n]:
            #             label[m * nCols + n] = i
            #             dist[m][n] = D
            # previous implementation end

        print("\t\t[{}] [DoSuperpixel.py]: K-means_iter_{}_step_2".format(time.ctime()[11:19], iteration))
        for i in range(seedNum):
            centerL1[i] = 0
            centerL2[i] = 0
            centera1[i] = 0
            centera2[i] = 0
            centerb1[i] = 0
            centerb2[i] = 0
            centerx1[i] = 0
            centerx2[i] = 0
            centery1[i] = 0
            centery2[i] = 0
            WSum[i] = 0
            clusterSize[i] = 0
            seedArray[i].x = 0
            seedArray[i].y = 0

        print("\t\t[{}] [DoSuperpixel.py]: K-means_iter_{}_step_3".format(time.ctime()[11:19], iteration))
        label = new_label.copy().reshape([nRows * nCols])
        # my implementation start : tested but slow (~= 17s)
        # step3_WL1 = W * L1
        # step3_WL2 = W * L2
        # step3_Wa1 = W * a1
        # step3_Wa2 = W * a2
        # step3_Wb1 = W * b1
        # step3_Wb2 = W * b2
        # step3_Wx1 = W * x1
        # step3_Wx2 = W * x2
        # step3_Wy1 = W * y1
        # step3_Wy2 = W * y2
        # for L in range(seedNum):
        #     if L % 50 == 0:
        #         print("\t\t\t\t[{}] [DEBUG]: seedNum{}".format(time.ctime()[11:19], L))
        #     add_range_matrix = (new_label == L)
        #     centerL1[L] += (step3_WL1 * add_range_matrix).sum()
        #     centerL2[L] += (step3_WL2 * add_range_matrix).sum()
        #     centera1[L] += (step3_Wa1 * add_range_matrix).sum()
        #     centera2[L] += (step3_Wa2 * add_range_matrix).sum()
        #     centerb1[L] += (step3_Wb1 * add_range_matrix).sum()
        #     centerb2[L] += (step3_Wb2 * add_range_matrix).sum()
        #     centerx1[L] += (step3_Wx1 * add_range_matrix).sum()
        #     centerx2[L] += (step3_Wx2 * add_range_matrix).sum()
        #     centery1[L] += (step3_Wy1 * add_range_matrix).sum()
        #     centery2[L] += (step3_Wy2 * add_range_matrix).sum()
        #     clusterSize[L] += add_range_matrix.sum()
        #     WSum[L] += (W * add_range_matrix).sum()
        #     seedArray[L].x, seedArray[L].y = np.sum(a=np.argwhere(add_range_matrix), axis=0)
        # my implementation end

        # previous implementation start (~= 10s)
        for i in range(nRows):
            for j in range(nCols):
                L = label[i * nCols + j]  # int
                Weight = W[i][j]  # double
                centerL1[L] += Weight * L1[i][j]
                centerL2[L] += Weight * L2[i][j]
                centera1[L] += Weight * a1[i][j]
                centera2[L] += Weight * a2[i][j]
                centerb1[L] += Weight * b1[i][j]
                centerb2[L] += Weight * b2[i][j]
                centerx1[L] += Weight * x1[i][j]
                centerx2[L] += Weight * x2[i][j]
                centery1[L] += Weight * y1[i][j]
                centery2[L] += Weight * y2[i][j]
                clusterSize[L] += 1
                WSum[L] += Weight
                seedArray[L].x += i
                seedArray[L].y += j
        # previous implementation end

        print("\t\t[{}] [DoSuperpixel.py]: K-means_iter_{}_step_4".format(time.ctime()[11:19], iteration))
        for i in range(seedNum):
            WSum[i] = 1 if WSum[i] == 0 else WSum[i]
            clusterSize[i] = 1 if clusterSize[i] == 0 else clusterSize[i]

        print("\t\t[{}] [DoSuperpixel.py]: K-means_iter_{}_step_5".format(time.ctime()[11:19], iteration))
        for i in range(seedNum):
            centerL1[i] /= WSum[i]
            centerL2[i] /= WSum[i]
            centera1[i] /= WSum[i]
            centera2[i] /= WSum[i]
            centerb1[i] /= WSum[i]
            centerb2[i] /= WSum[i]
            centerx1[i] /= WSum[i]
            centerx2[i] /= WSum[i]
            centery1[i] /= WSum[i]
            centery2[i] /= WSum[i]
            seedArray[i].x /= clusterSize[i]
            seedArray[i].y /= clusterSize[i]

    if FAKE_KMEANS_LABEL:
        label = pickle.load(open("test_dump_data\\test_27_DOS_label_iter20.pydump", 'rb'))
    if TEST_KMEANS_LABEL:
        data = loadmat("test_matlab_data\\test_27_DOS_label_after_KMEANS.mat")
        print(compare_matrix.compare_2D_matrix(label.reshape([nRows, nCols]).transpose([1, 0]), data["tLabel"]))
        exit()

    threshold = int((nRows * nCols) / (seedNum * thresholdCoef))
    preEnforceConnectivity(label, nRows, nCols)

    if TEST_PEC_LABEL:
        data = loadmat("test_matlab_data\\test_27_DOS_label_after_PEC.mat")
        print(compare_matrix.compare_2D_matrix(label.reshape([nRows, nCols]).transpose([1, 0]), data["tLabel"]))
        exit()

    if FAKE_EC_LABEL:
        label = loadmat("test_matlab_data\\FINAL.mat")["label"].transpose([1, 0]).reshape([nRows * nCols])
    else:
        label = EnforceConnectivity(L1, L2, a1, a2, b1, b2, x1, x2, y1, y2, W, label, threshold, nRows, nCols)
    return label
