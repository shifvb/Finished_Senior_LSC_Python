# Enforce Connectivity by merging very small superpixels with their neighbors
import numpy as np
import time


def preEnforceConnectivity(label: np.ndarray, nRows: int, nCols: int):
    print("[{}] preEnforceConnectivity...".format(time.ctime()[11:19]))
    dx8 = (-1, -1, 0, 1, 1, 1, 0, -1)
    dy8 = (0, -1, -1, -1, 0, 1, 1, 1)
    adj = 0
    Bond = 20
    mask = np.zeros([nRows, nCols], dtype=np.bool)  # bool type(C++)
    xLoc = []
    yLoc = []
    for i in range(nRows):
        for j in range(nCols):
            if mask[i][j] == 0:
                L = label[i * nCols + j]
                for k in range(8):
                    x = i + dx8[k]
                    y = j + dy8[k]
                    if 0 <= x <= nRows - 1 and 0 <= y <= nCols - 1:
                        if mask[x][y] and label[x * nCols + y] != L:
                            adj = label[x * nCols + y]
                            break
                mask[i][j] = 1
                xLoc.append(i)
                yLoc.append(j)
                indexMarker = 0
                while indexMarker < len(xLoc):
                    x = xLoc[indexMarker]
                    y = yLoc[indexMarker]
                    indexMarker += 1
                    minX = 0 if x - 1 <= 0 else x - 1
                    maxX = nRows - 1 if x + 1 >= nRows - 1 else x + 1
                    minY = 0 if y - 1 <= 0 else y - 1
                    maxY = nCols - 1 if y + 1 >= nCols - 1 else y + 1
                    for m in range(minX, maxX + 1):
                        for n in range(minY, maxY + 1):
                            if not mask[m][n] and label[m * nCols + n] == L:
                                mask[m][n] = 1
                                xLoc.append(m)
                                yLoc.append(n)
                if indexMarker < Bond:
                    for k in range(len(xLoc)):
                        x = xLoc[k]
                        y = yLoc[k]
                        label[x * nCols + y] = adj
                xLoc.clear()
                yLoc.clear()
    del mask
