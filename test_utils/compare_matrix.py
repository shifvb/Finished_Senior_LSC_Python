from scipy.io import loadmat
from math import fabs


def compare_3D_matrix(python_array,
                      matlab_array,
                      max_count_num=100,
                      diff_threshold=0):
    A = python_array
    B = matlab_array
    assert A.shape == B.shape, "A:{}, B:{}".format(A.shape, B.shape)
    assert A.dtype == B.dtype, "A:{}, B:{}".format(A.dtype, B.dtype)

    count = 0
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            for k in range(A.shape[2]):
                if count > max_count_num:
                    return count
                diff = A[r][c][k] - B[r][c][k] if A[r][c][k] > B[r][c][k] else B[r][c][k] - A[r][c][k]
                if diff > diff_threshold:
                    count += 1
                    print("[DEBUG] pos: {}, A:{}, B: {}, diff:{}".format((r, c, k), A[r][c][k], B[r][c][k], diff))
    return count


def compare_2D_matrix(python_array,
                      matlab_array,
                      max_count_num=100,
                      diff_threshold=0):
    A = python_array
    B = matlab_array
    assert A.shape == B.shape, "A:{}, B:{}".format(A.shape, B.shape)
    assert A.dtype == B.dtype, "A:{}, B:{}".format(A.dtype, B.dtype)

    count = 0
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            if count > max_count_num:
                return count
            diff = A[r][c] - B[r][c] if A[r][c] > B[r][c] else B[r][c] - A[r][c]
            if diff > diff_threshold:
                count += 1
                print("[DEBUG] pos: {}, A:{}, B: {}, diff:{}".format((r, c), A[r][c], B[r][c], diff))
    return count


def compare_1D_array(python_array, matlab_array, max_count_num=100, diff_threshold=0):
    A = python_array
    B = matlab_array
    assert A.dtype == B.dtype, "A:{}, B:{}".format(A.dtype, B.dtype)
    assert A.shape == B.shape, "A:{}, B:{}".format(A.shape, B.shape)
    count = 0
    for i in range(A.shape[0]):
        if count > max_count_num:
            return count
        diff = A[i] - B[i] if A[i] > B[i] else B[i] - A[i]
        if diff > diff_threshold:
            count += 1
            print("[DEBUG] pos: {}, A:{}, B: {}, diff:{}".format(i, A[i], B[i], diff))
    return count
