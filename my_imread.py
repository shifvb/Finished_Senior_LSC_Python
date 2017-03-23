from PIL import Image
import numpy as np


# def my_imread_deprecated(path):
#     import cv2
#     arr = cv2.imread(path)
#     for i in range(arr.shape[0]):  # change BGR to RGB image
#         for j in range(arr.shape[1]):
#             t = arr[i][j][2]
#             arr[i][j][2] = arr[i][j][0]
#             arr[i][j][0] = t
#     return arr


def my_imread(path: str):
    return np.asarray(Image.open(path), dtype=np.uint8)
