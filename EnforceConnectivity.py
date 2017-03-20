import numpy as np
from queue import Queue
import sys
import time
from scipy.io import loadmat
from test_utils import compare_matrix

DBL_MAX = sys.float_info[0]  # max float value

TEST_LABEL_STEP_1 = False
TEST_LABEL_STEP_2 = False


class Superpixel(object):
    def __init__(self, L=0, S=0):
        self.Label = L
        self.Size = S
        self.xLoc = []
        self.yLoc = []
        self.Neighbor = []

    def __eq__(self, other):
        if isinstance(other, int):
            raise NotImplementedError("Not implemented!(int)")
        elif isinstance(other, Superpixel):
            return self.Label == other.Label
        else:
            raise NotImplementedError("Compare between class<Superpixel> and other classes not implemented yet!")

    def __str__(self):
        return "<class 'Superpixel'> label={}, Size={}, xLoc={}, yLoc={}, Neighbor={}".format(self.Label, self.Size,
                                                                                              self.xLoc, self.yLoc,
                                                                                              self.Neighbor)

    __repr__ = __str__


def EnforceConnectivity(L1: np.ndarray, L2: np.ndarray, a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray,
                        x1: np.ndarray, x2: np.ndarray, y1: np.ndarray, y2: np.ndarray, W: np.ndarray,
                        label: np.ndarray, threshold: int, nRows: int, nCols: int):
    print("[{}] EnforceConnectivity...".format(time.ctime()[11:19]))
    print("\t[{}] [EnforceConnectivity.py] step_1/3".format(time.ctime()[11:19]))
    mask = np.zeros([nRows, nCols], dtype=np.bool)
    strayX = []  # unsigned short
    strayY = []  # unsigned short
    Size = []  # unsigned short
    xLoc = []  # unsigned short
    yLoc = []  # unsigned short
    centerL1 = []  # double
    centerL2 = []  # double
    centera1 = []  # double
    centera2 = []  # double
    centerb1 = []  # double
    centerb2 = []  # double
    centerx1 = []  # double
    centerx2 = []  # double
    centery1 = []  # double
    centery2 = []  # double
    centerW = []  # double

    sLabel = -1  # int

    for i in range(nRows):
        for j in range(nCols):
            if mask[i][j] == 0:
                sLabel += 1
                Count = 1
                centerL1.append(0)
                centerL2.append(0)
                centera1.append(0)
                centera2.append(0)
                centerb1.append(0)
                centerb2.append(0)
                centerx1.append(0)
                centerx2.append(0)
                centery1.append(0)
                centery2.append(0)
                centerW.append(0)
                strayX.append(i)
                strayY.append(j)
                Weight = W[i][j]  # double
                centerL1[sLabel] += L1[i][j] * Weight
                centerL2[sLabel] += L2[i][j] * Weight
                centera1[sLabel] += a1[i][j] * Weight
                centera2[sLabel] += a2[i][j] * Weight
                centerb1[sLabel] += b1[i][j] * Weight
                centerb2[sLabel] += b2[i][j] * Weight
                centerx1[sLabel] += x1[i][j] * Weight
                centerx2[sLabel] += x2[i][j] * Weight
                centery1[sLabel] += y1[i][j] * Weight
                centery2[sLabel] += y2[i][j] * Weight
                centerW[sLabel] += W[i][j]
                L = label[i * nCols + j]
                label[i * nCols + j] = sLabel
                mask[i][j] = 1
                xLoc.append(i)
                yLoc.append(j)
                while len(xLoc) > 0:
                    x = xLoc.pop(0)
                    y = yLoc.pop(0)
                    minX = 0 if x - 1 <= 0 else x - 1
                    maxX = nRows - 1 if x + 1 >= nRows - 1 else x + 1
                    minY = 0 if y - 1 <= 0 else y - 1
                    maxY = nCols - 1 if y + 1 >= nCols - 1 else y + 1
                    for m in range(minX, maxX + 1):
                        for n in range(minY, maxY + 1):
                            if not mask[m][n] and label[m * nCols + n] == L:
                                Count += 1
                                xLoc.append(m)
                                yLoc.append(n)
                                mask[m][n] = 1
                                label[m * nCols + n] = sLabel
                                Weight = W[m][n]
                                centerL1[sLabel] += L1[m][n] * Weight
                                centerL2[sLabel] += L2[m][n] * Weight
                                centera1[sLabel] += a1[m][n] * Weight
                                centera2[sLabel] += a2[m][n] * Weight
                                centerb1[sLabel] += b1[m][n] * Weight
                                centerb2[sLabel] += b2[m][n] * Weight
                                centerx1[sLabel] += x1[m][n] * Weight
                                centerx2[sLabel] += x2[m][n] * Weight
                                centery1[sLabel] += y1[m][n] * Weight
                                centery2[sLabel] += y2[m][n] * Weight
                                centerW[sLabel] += W[m][n]
                Size.append(Count)
                centerL1[sLabel] /= centerW[sLabel]
                centerL2[sLabel] /= centerW[sLabel]
                centera1[sLabel] /= centerW[sLabel]
                centera2[sLabel] /= centerW[sLabel]
                centerb1[sLabel] /= centerW[sLabel]
                centerb2[sLabel] /= centerW[sLabel]
                centerx1[sLabel] /= centerW[sLabel]
                centerx2[sLabel] /= centerW[sLabel]
                centery1[sLabel] /= centerW[sLabel]
                centery2[sLabel] /= centerW[sLabel]
    sLabel += 1
    Count = 0

    if TEST_LABEL_STEP_1:
        data = loadmat("test_matlab_data\\test_27_EC_label_step1.mat")
        print(
            compare_matrix.compare_2D_matrix(label.reshape([nRows, nCols]), data["tLabel"].transpose([1, 0]), 1000, 0))
        exit()

    print("\t[{}] [EnforceConnectivity.py] step_2/3".format(time.ctime()[11:19]))
    Sarray = []  # vector<Superpixel> Sarray;
    for i in range(sLabel):
        if Size[i] < threshold:
            x = strayX[i]
            y = strayY[i]
            L = label[x * nCols + y]
            mask[x][y] = 0
            indexMark = 0
            S = Superpixel(L, Size[i])
            S.xLoc.append(x)
            S.yLoc.append(y)
            while indexMark < len(S.xLoc):
                x = S.xLoc[indexMark]
                y = S.yLoc[indexMark]
                indexMark += 1
                minX = 0 if x - 1 <= 0 else x - 1
                maxX = nRows - 1 if x + 1 >= nRows - 1 else x + 1
                minY = 0 if y - 1 <= 0 else y - 1
                maxY = nCols - 1 if y + 1 >= nCols - 1 else y + 1
                for m in range(minX, maxX + 1):
                    for n in range(minY, maxY + 1):
                        if mask[m][n] and label[m * nCols + n] == L:
                            mask[m][n] = 0
                            S.xLoc.append(m)
                            S.yLoc.append(n)
                        elif label[m * nCols + n] != L:
                            NewLabel = label[m * nCols + n]
                            if NewLabel not in S.Neighbor:
                                S.Neighbor.insert(0, NewLabel)
            Sarray.append(S)

    if TEST_LABEL_STEP_2:
        data = loadmat("test_matlab_data\\test_27_EC_label_step2.mat")
        print(
            compare_matrix.compare_2D_matrix(label.reshape([nRows, nCols]), data["tLabel"].transpose([1, 0]), 1000, 0))
        exit()

    print("\t[{}] [EnforceConnectivity.py] step_3/3".format(time.ctime()[11:19]))
    S = 0
    while len(Sarray) > 0:
        MinDist = DBL_MAX
        Label1 = int(Sarray[S].Label)
        Label2 = -1
        for I in range(len(Sarray[S].Neighbor)):
            D = (centerL1[Label1] - centerL1[Sarray[S].Neighbor[I]]) * (centerL1[Label1] - centerL1[Sarray[S].Neighbor[I]]) + \
                (centerL2[Label1] - centerL2[Sarray[S].Neighbor[I]]) * (centerL2[Label1] - centerL2[Sarray[S].Neighbor[I]]) + \
                (centera1[Label1] - centera1[Sarray[S].Neighbor[I]]) * (centera1[Label1] - centera1[Sarray[S].Neighbor[I]]) + \
                (centera2[Label1] - centera2[Sarray[S].Neighbor[I]]) * (centera2[Label1] - centera2[Sarray[S].Neighbor[I]]) + \
                (centerb1[Label1] - centerb1[Sarray[S].Neighbor[I]]) * (centerb1[Label1] - centerb1[Sarray[S].Neighbor[I]]) + \
                (centerb2[Label1] - centerb2[Sarray[S].Neighbor[I]]) * (centerb2[Label1] - centerb2[Sarray[S].Neighbor[I]]) + \
                (centerx1[Label1] - centerx1[Sarray[S].Neighbor[I]]) * (centerx1[Label1] - centerx1[Sarray[S].Neighbor[I]]) + \
                (centerx2[Label1] - centerx2[Sarray[S].Neighbor[I]]) * (centerx2[Label1] - centerx2[Sarray[S].Neighbor[I]]) + \
                (centery1[Label1] - centery1[Sarray[S].Neighbor[I]]) * (centery1[Label1] - centery1[Sarray[S].Neighbor[I]]) + \
                (centery2[Label1] - centery2[Sarray[S].Neighbor[I]]) * (centery2[Label1] - centery2[Sarray[S].Neighbor[I]])
            if abs(D - MinDist) > 1e-6:
                MinDist = D
                Label2 = Sarray[S].Neighbor[I]
        W1 = centerW[Label1]
        W2 = centerW[Label2]
        W = W1 + W2
        centerL1[Label2] = (W2 * centerL1[Label2] + W1 * centerL1[Label1]) / W
        centerL2[Label2] = (W2 * centerL2[Label2] + W1 * centerL2[Label1]) / W
        centera1[Label2] = (W2 * centera1[Label2] + W1 * centera1[Label1]) / W
        centera2[Label2] = (W2 * centera2[Label2] + W1 * centera2[Label1]) / W
        centerb1[Label2] = (W2 * centerb1[Label2] + W1 * centerb1[Label1]) / W
        centerb2[Label2] = (W2 * centerb2[Label2] + W1 * centerb2[Label1]) / W
        centerx1[Label2] = (W2 * centerx1[Label2] + W1 * centerx1[Label1]) / W
        centerx2[Label2] = (W2 * centerx2[Label2] + W1 * centerx2[Label1]) / W
        centery1[Label2] = (W2 * centery1[Label2] + W1 * centery1[Label1]) / W
        centery2[Label2] = (W2 * centery2[Label2] + W1 * centery2[Label1]) / W
        centerW[Label2] = W

        for i in range(len(Sarray[S].xLoc)):
            x = Sarray[S].xLoc[i]
            y = Sarray[S].yLoc[i]
            label[x * nCols + y] = Label2

        if Superpixel(Label2) in Sarray:
            Stmp = Sarray.index(Superpixel(Label2))
            Size[Label2] = Size[Label1] + Size[Label2]
            if Size[Label2] >= threshold:
                del Sarray[Stmp]
                del Sarray[S]
            else:
                Sarray[Stmp].xLoc.extend(Sarray[S].xLoc)
                Sarray[Stmp].yLoc.extend(Sarray[S].yLoc)
                Sarray[Stmp].Neighbor.extend(Sarray[S].Neighbor)
                Sarray[Stmp].Neighbor = list(set(Sarray[Stmp].Neighbor))
                Sarray[Stmp].Neighbor.sort()
                I = Sarray[Stmp].Neighbor.index(Label1)
                del Sarray[Stmp].Neighbor[I]
                I = Sarray[Stmp].Neighbor.index(Label2)
                del Sarray[Stmp].Neighbor[I]
                del Sarray[S]
        else:
            del Sarray[S]

        for i in range(len(Sarray)):
            if Label1 in Sarray[i].Neighbor and Label2 in Sarray[i].Neighbor:
                I = Sarray[i].Neighbor.index(Label1)
                del Sarray[i].Neighbor[I]
            elif Label1 in Sarray[i].Neighbor and Label2 not in Sarray[i].Neighbor:
                I = Sarray[i].Neighbor.index(Label1)
                Sarray[i].Neighbor[I] = Label2
        S = 0
    return label
