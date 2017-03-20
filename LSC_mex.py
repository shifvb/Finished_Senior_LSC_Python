import numpy as np
from LSC import LSC


def LSC_mex(I, superpixelNum: int, ratio: float):
    """
    mex wrapper of mexFuncion
    :param I: image data
    :param superpixelNum: number of superpixel to generate
    :param ratio: I don't know the algorithm... just ratio
    :return: label data
    """
    return mexFunction(I, superpixelNum, ratio)


def mexFunction(I: np.ndarray, superpixelNum: int, ratio=0.1):
    assert len(I.shape) == 3, "The input image must be in CIERGB form"
    assert I.dtype == np.uint8, "The input image must be in CIERGB form"
    nRows, nCols, _ = I.shape
    pixel = nRows * nCols
    label = np.empty([pixel], dtype=np.uint16)
    return LSC(I, nCols, nRows, superpixelNum, ratio, label)
