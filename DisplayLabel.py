import numpy as np
from PIL import Image


def DisplayLabel(label_2D: np.ndarray, name):
    nRows, nCols = label_2D.shape
    img = np.zeros([nRows, nCols, 3], dtype=np.uint8)
    for i in range(nRows):
        for j in range(nCols):
            minX = 0 if i - 1 < 0 else i - 1
            minY = 0 if j - 1 < 0 else j - 1
            maxX = nRows - 1 if i + 1 >= nRows else i + 1
            maxY = nCols - 1 if j + 1 >= nCols else j + 1
            count = (label_2D[minX:maxX + 1, minY:maxY + 1] != label_2D[i][j]).sum()
            if count >= 2:
                img[i][j] = [255, 255, 255]
    PIL_image = Image.fromarray(img, 'RGB')
    PIL_image.show()
    PIL_image.save(name.split(".")[0] + "_label" + ".jpg")
