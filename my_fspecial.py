import numpy as np


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def my_fspecial(type_str, shape, *args):
    if type_str != "gaussian":
        raise ValueError("type {} not implemented!".format(type_str))
    if isinstance(shape, int):
        shape = [shape, shape]
    if len(args) == 0:
        return matlab_style_gauss2D(shape)
    return matlab_style_gauss2D(shape, args[0])


def test():
    r = my_fspecial('gaussian', 5)
    print(r)

if __name__ == '__main__':
    test()
