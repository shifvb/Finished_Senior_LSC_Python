import numpy as np
from scipy.ndimage import correlate
from my_imread import my_imread
from my_fspecial import my_fspecial
from LSC_mex import LSC_mex
from test_utils import compare_matrix
from PIL import Image
from scipy.io import loadmat
from DisplayLabel import DisplayLabel
from DislaySuperpixel import DisplaySuperpixel


# [MATLAB] name='02';
name = '27ab22bdff8437f2dab2f16849abf938.jpg'
# [MATLAB] img=imread([name,'.jpg']);
img = my_imread(name)
# [MATLAB] gaus=fspecial('gaussian',3);
gaus = my_fspecial('gaussian', 3)
# [MATLAB] I=imfilter(img,gaus);
I = correlate(img.astype(np.float64),
              gaus.reshape(gaus.shape[0], gaus.shape[1], 1),
              mode="constant").round().astype(np.uint8)
# [MATLAB] superpixelNum=200;
superpixelNum = 200
# [MATLAB] ratio=0.075;
ratio = 0.075
# [MATLAB] label=LSC_mex(I,superpixelNum,ratio);
label = LSC_mex(I, superpixelNum, ratio)

# label_2D = loadmat("FINAL.mat")["label"]
nRows, nCols, _ = img.shape
label_2D = label.reshape([nCols, nRows]).transpose([1, 0])

# [MATLAB]DisplaySuperpixel(label,img,name);
DisplaySuperpixel(label_2D, img, name)
# [MATLAB]DisplayLabel(label,name);
DisplayLabel(label_2D, name)
