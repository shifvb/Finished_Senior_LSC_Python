from math import pow
import numpy as np
import math
from skimage import color


# # Change from RGB colour space to LAB colour space
# def RGB2XYZ_deprecated_v1(sR: int, sG: int, sB: int):
#     R = sR / 255.0
#     G = sG / 255.0
#     B = sB / 255.0
#
#     r = R / 12.92 if R <= 0.04045 else math.pow((R + 0.055) / 1.055, 2.4)
#     g = G / 12.92 if G <= 0.04045 else math.pow((G + 0.055) / 1.055, 2.4)
#     b = B / 12.92 if B <= 0.04045 else math.pow((B + 0.055) / 1.055, 2.4)
#
#     X = r * 0.412453 + g * 0.357580 + b * 0.180423
#     Y = r * 0.212671 + g * 0.715160 + b * 0.072169
#     Z = r * 0.019334 + g * 0.119193 + b * 0.950227
#     return X, Y, Z
#
#
# def RGB2LAB_deprecated_v1(sR: int, sG: int, sB: int):
#     X, Y, Z = RGB2XYZ_deprecated_v1(sR, sG, sB)
#
#     epsilon = 0.008856  # actual CIE standard
#     kappa = 903.3  # actual CIE standard
#
#     Xr = 0.950456  # reference white
#     Yr = 1.0  # reference white
#     Zr = 1.088754  # reference white
#
#     xr = X / Xr
#     yr = Y / Yr
#     zr = Z / Zr
#
#     fx = math.pow(xr, 1.0 / 3.0) if xr > epsilon else (kappa * xr + 16.0) / 116.0
#     fy = math.pow(yr, 1.0 / 3.0) if yr > epsilon else (kappa * yr + 16.0) / 116.0
#     fz = math.pow(zr, 1.0 / 3.0) if zr > epsilon else (kappa * zr + 16.0) / 116.0
#
#     lval = (116.0 * fy - 16.0) / 100 * 255 + 0.5
#     aval = 500.0 * (fx - fy) + 128 + 0.5
#     bval = 200.0 * (fy - fz) + 128 + 0.5
#     return int(lval), int(aval), int(bval)


# def myrgb2lab_deprecated(L: np.ndarray, A: np.ndarray, B: np.ndarray, nRows: int, nCols: int, I: np.ndarray) -> None:
#     epsilon = 0.008856  # actual CIE standard
#     kappa = 903.3  # actual CIE standard
#     xyz_data = color.rgb2xyz(I).transpose([2, 1, 0])
#     xr = xyz_data[0].copy().reshape([nRows * nCols]) / 0.950456
#     yr = xyz_data[1].copy().reshape([nRows * nCols])
#     zr = xyz_data[2].copy().reshape([nRows * nCols]) / 1.088754
#     for i in range(nCols * nRows):
#         temp_xr = xr[i]
#         temp_yr = yr[i]
#         temp_zr = zr[i]
#         fx = pow(temp_xr, 1.0 / 3.0) if temp_xr > epsilon else (kappa * temp_xr + 16.0) / 116.0
#         fy = pow(temp_yr, 1.0 / 3.0) if temp_yr > epsilon else (kappa * temp_yr + 16.0) / 116.0
#         fz = pow(temp_zr, 1.0 / 3.0) if temp_zr > epsilon else (kappa * temp_zr + 16.0) / 116.0
#         L[i] = (116.0 * fy - 16.0) / 100 * 255 + 0.5
#         A[i] = 500.0 * (fx - fy) + 128 + 0.5
#         B[i] = 200.0 * (fy - fz) + 128 + 0.5


def myrgb2lab(I: np.ndarray, row_num: int, col_num: int):
    """
    change rgb to lab format
    :param I: rgb format image
    :return:
        L: L channel, range from 0 to 255, dtype: uint8, shape: (row_num * col_num,)
        a: a channel, range from 0 to 255, dtype: uint8, shape: (row_num * col_num,)
        b: b channel, range from 0 to 255, dtype: uint8, shape: (row_num * col_num,)
    """
    lab_img = color.rgb2lab(I).transpose([2, 1, 0])
    L = lab_img[0].copy().reshape([row_num * col_num])
    a = lab_img[1].copy().reshape([row_num * col_num])
    b = lab_img[2].copy().reshape([row_num * col_num])
    L /= (100 / 255)  # L is [0, 100], change it to [0, 255]
    L += 0.5
    a += 128 + 0.5  # A is [-128, 127], change it to [0, 255]
    b += 128 + 0.5  # B is [-128, 127], change it to [0, 255]
    return L.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)
