# Linear-Spectral-Clustering-Superpixel-Segmentation-Algorithm_Python
A Python implementation of LSC algorithm by shifvb

Developed on Python3.6(Windows x86_64), should be run well on Python3.3+

If you want to see the demo, just run LSC_demo.py

## (C) Zhengqin Li, Jiansheng Chen, 2014
[HomePage](http://jschenthu.weebly.com/projects.html)

You are free to use, change or redistribute this code for any non-commrecial purposes.
If you use this software,please cite thefollowing in any resulting publication and email us:

[1] Zhengqin Li, Jiansheng Chen, Superpixel Segmentation using Linear Spectral Clustering, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2015

(C) Zhengqin Li, Jiansheng Chen, 2014

li-zq12@mails.tsinghua.edu.cn

jschenthu@mail.tsinghua.edu.cn

Tsinghua University

### Abstract

![](http://jschenthu.weebly.com/uploads/2/4/1/1/24110356/6091384_1_orig.jpg)
 
We present in this paper a superpixel segmentation algorithm called Linear Spectral Clustering (LSC), which produces compact and uniform superpixels with low computational costs.
Basically, a normalized cuts formulation of the superpixel segmentation is adopted based on a similarity metric that measures the color similarity and space proximity between image pixels.
However, instead of using the traditional eigen-based algorithm, we approximate the similarity metric using a kernel function leading to an explicitly mapping of pixel values and coordinates into a high dimensional feature space.
We prove that by appropriately weighting each point in this feature space, the objective functions of weighted K-means and normalized cuts share the same optimum point.
As such, it is possible to optimize the cost function of normalized cuts by iteratively applying simple K-means clustering in the proposed feature space.
LSC is of linear computational complexity and high memory efficiency and is able to preserve global properties of images.
Experimental results show that LSC performs equally well or better than state of the art superpixel segmentation algorithms in terms of several commonly used evaluation metrics in image segmentation.

### Grant:

National Natural Science Foundation of China Project (#61101152)

Tsinghua University Initiative Scientific Research Program Project (#20131089382)

Beijing Higher Education Young Elite Teacher Project (#YETP0104)

### Publication:

Zhengqin Li, Jiansheng Chen, Superpixel Segmentation using Linear Spectral Clustering, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2015



