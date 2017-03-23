from myrgb2lab import myrgb2lab
import cmath
from Seeds import gen_seeds
from Initialize import Initialize
import numpy as np
from DoSuperpixel import DoSuperpixel
from scipy.io import loadmat
from test_utils import compare_matrix
from test_utils.compare_matrix import compare_2D_matrix
import time

TEST_RGB2LAB = False
TEST_INITIALIZATION = False
FAKE_INITIALIZATION = False


# LSC superpixel segmentation algorithm
def LSC(I: np.ndarray, nRows: int, nCols: int, superpixelnum: int, ratio: float, label: np.ndarray):
    new_label = np.empty([nRows, nCols], dtype=np.uint16)
    print("[{}] Setting Parameter...".format(time.ctime()[11:19]))
    colorCoefficient = 20
    distCoefficient = colorCoefficient * ratio
    seedNum = superpixelnum
    iterationNum = 20
    thresholdCoef = 4

    print("[{}] Translating image from RGB format to LAB format...".format(time.ctime()[11:19]))
    # img = I.transpose([2, 1, 0])
    # R = np.copy(img[0]).reshape([nRows * nCols])
    # G = np.copy(img[1]).reshape([nRows * nCols])
    # B = np.copy(img[2]).reshape([nRows * nCols])
    # L = np.empty([nRows * nCols], dtype=np.uint8)
    # a = np.empty([nRows * nCols], dtype=np.uint8)
    # b = np.empty([nRows * nCols], dtype=np.uint8)
    # myrgb2lab(L, a, b, nRows, nCols, I)
    L, a, b = myrgb2lab(I, nRows, nCols)

    if TEST_RGB2LAB:
        data = loadmat(r"test_matlab_data\test_27_RGBLAB.mat")
        print(compare_2D_matrix(R.copy().reshape([nRows, nCols]).transpose([1, 0]), data["tR"], 10, 1), end="", flush=True)
        print(compare_2D_matrix(G.copy().reshape([nRows, nCols]).transpose([1, 0]), data["tG"], 10, 1), end="", flush=True)
        print(compare_2D_matrix(B.copy().reshape([nRows, nCols]).transpose([1, 0]), data["tB"], 10, 1), end="", flush=True)
        print(compare_2D_matrix(L.copy().reshape([nRows, nCols]).transpose([1, 0]), data["tL"], 10, 1), end="", flush=True)
        print(compare_2D_matrix(a.copy().reshape([nRows, nCols]).transpose([1, 0]), data["ta"], 10, 1), end="", flush=True)
        print(compare_2D_matrix(b.copy().reshape([nRows, nCols]).transpose([1, 0]), data["tb"], 10, 1), end="", flush=True)
        exit()

    print("[{}] Producing Seeds...".format(time.ctime()[11:19]))
    ColNum = int(cmath.sqrt(seedNum * nCols / nRows).real)
    RowNum = int(seedNum / ColNum)
    StepX = int(nRows / RowNum)
    StepY = int(nCols / ColNum)
    # seedArray = []
    # newSeedNum = Seeds_deprecated(nRows, nCols, RowNum, ColNum, StepX, StepY, seedNum, seedArray)
    seedArray = gen_seeds(row_num=nRows, col_num=nCols, seed_num=seedNum)
    newSeedNum = len(seedArray)

    print("[{}] Initialization...".format(time.ctime()[11:19]))
    L1, L2, a1, a2, b1, b2, x1, x2, y1, y2, W = Initialize(L, a, b, nRows, nCols, StepX, StepY, colorCoefficient,
                                                           distCoefficient)

    if FAKE_INITIALIZATION:
        data = loadmat("test_matlab_data\\test_27_Init.mat")
        L1 = data["tL1"]
        L2 = data["tL2"]
        a1 = data["ta1"]
        a2 = data["ta2"]
        b1 = data["tb1"]
        b2 = data["tb2"]
        x1 = data["tx1"]
        x2 = data["tx2"]
        y1 = data["ty1"]
        y2 = data["ty2"]
        W = data["tW"]
    if TEST_INITIALIZATION:
        data = loadmat("test_matlab_data\\test_27_Init.mat")
        print(compare_matrix.compare_2D_matrix(L1, data["tL1"], 10, 1e-4), end="", flush=True)
        print(compare_matrix.compare_2D_matrix(L2, data["tL2"], 10, 1e-4), end="", flush=True)
        print(compare_matrix.compare_2D_matrix(a1, data["ta1"], 10, 1e-4), end="", flush=True)
        print(compare_matrix.compare_2D_matrix(a2, data["ta2"], 10, 1e-4), end="", flush=True)
        print(compare_matrix.compare_2D_matrix(b1, data["tb1"], 10, 1e-4), end="", flush=True)
        print(compare_matrix.compare_2D_matrix(b2, data["tb2"], 10, 1e-4), end="", flush=True)
        print(compare_matrix.compare_2D_matrix(x1, data["tx1"], 10, 1e-4), end="", flush=True)
        print(compare_matrix.compare_2D_matrix(x2, data["tx2"], 10, 1e-4), end="", flush=True)
        print(compare_matrix.compare_2D_matrix(y1, data["ty1"], 10, 1e-4), end="", flush=True)
        print(compare_matrix.compare_2D_matrix(y2, data["ty2"], 10, 1e-4), end="", flush=True)
        print(compare_matrix.compare_2D_matrix(W, data["tW"], 10, 1e-1), end="\n", flush=True)
        exit()

    del L
    del a
    del b

    print("[{}] Producing Superpixel...".format(time.ctime()[11:19]))
    return DoSuperpixel(L1, L2, a1, a2, b1, b2, x1, x2, y1, y2, W, label, seedArray, newSeedNum, nRows, nCols, StepX,
                        StepY, iterationNum, thresholdCoef, new_label)
