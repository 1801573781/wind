"""
Function：卷积测试
Author：lzb
Date：2021.01.09
"""

import numpy as np
import os

from cnn.max_pooling import MaxPooling
from cnn.mean_pooling import MeanPooling
from cnn.convolution import Convolution, Reversal, ConvolutionType
from my_image.image import show_file, gray_file, show_data, ImageDataType, get_data

"""
功能：测试卷积网络-2维
参数：NULL 
返回值：NULL
"""


def test_cvl_2d():
    # 输入信息 x
    x = np.asarray([[1, 1, 1, 1, 1],
                    [-1, 0, -3, 0, 1],
                    [2, 1, 1, -1, 0],
                    [0, -1, 1, 2, 1],
                    [1, 2, 1, 1, 1]])

    # 滤波器 w
    w = np.asarray([[1, 0, 0],
                    [0, 0, 0],
                    [0, 0, -1]])

    con = Convolution()

    y, err = con.convolution(w, x, Reversal.REV, ConvolutionType.Other, 2, 10, 10)

    print("\n")

    print(y)


"""
功能：测试卷积网络-3维
参数：NULL 
返回值：NULL
"""


def test_cvl_3d():
    # 输入信息 x
    x = np.asarray([[[1], [1], [1], [1], [1]],
                    [[-1], [0], [-3], [0], [1]],
                    [[2], [1], [1], [-1], [0]],
                    [[0], [-1], [1], [2], [1]],
                    [[1], [2], [1], [1], [1]]])

    # 滤波器 w
    w = np.asarray([[[1], [0], [0]],
                    [[0], [0], [0]],
                    [[0], [0], [-1]]])

    con = Convolution()

    y, err = con.convolution(w, x, Reversal.REV, ConvolutionType.Other, 2, 10, 10)

    print("\n")

    print(y)


"""
功能：测试卷积网络-图像-2维
参数：NULL 
返回值：NULL
"""


def test_image_2d():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # 图像数据（灰度）
    file_name = "./../my_image/dog1.bmp"

    show_file(file_name)

    gray, err = gray_file(file_name)
    show_data(gray, ImageDataType.GRAY)

    # 卷积
    # con = Convolution()
    con = MeanPooling()

    # 输入信息 x
    x = gray

    # 滤波器 w（平滑去噪）
    w = np.asarray([[1 / 16, 1 / 8, 1 / 16],
                    [1 / 8, 1 / 4, 1 / 8],
                    [1 / 16, 1 / 8, 1 / 16]])

    y, err = con.convolution(w, x, Reversal.REV, ConvolutionType.Other, 3, 0, 0)

    show_data(y, ImageDataType.GRAY)

    # 滤波器 w（提取边缘特征）
    w = np.asarray([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])

    y, err = con.convolution(w, x, Reversal.REV, ConvolutionType.Other, 3, 0, 0)

    show_data(y, ImageDataType.GRAY)

    # 滤波器 w（提取边缘特征）
    w = np.asarray([[0, 1, 1],
                    [-1, 0, 1],
                    [-1, -1, 0]])

    y, err = con.convolution(w, x, Reversal.REV, ConvolutionType.Other, 3, 0, 0)

    show_data(y, ImageDataType.GRAY)


"""
功能：测试卷积网络-图像-3维
参数：NULL 
返回值：NULL
"""


def test_image_3d():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # 图像数据
    file_name = "./../my_image/dog1.bmp"

    show_file(file_name)

    data, image_data_type, err = get_data(file_name)

    # 卷积
    # con = Convolution()
    con = MeanPooling()

    # 输入信息 x
    x = data

    # 滤波器 w（平滑去噪）
    w = np.asarray([[[1 / 16, 1 / 16, 1 / 16], [1 / 8, 1 / 8, 1 / 8], [1 / 16, 1 / 16, 1 / 16]],
                    [[1 / 8, 1 / 8, 1 / 8], [1 / 4, 1 / 4, 1 / 4], [1 / 8, 1 / 8, 1 / 8]],
                    [[1 / 16, 1 / 16, 1 / 16], [1 / 8, 1 / 8, 1 / 8], [1 / 16, 1 / 16, 1 / 16]]])

    y, err = con.convolution(w, x, Reversal.REV, ConvolutionType.Other, 3, 0, 0)

    show_data(y, ImageDataType.RGB)

    # 滤波器 w（提取边缘特征）
    w = np.asarray([[[0, 0, 0], [1, 1, 1], [0, 0, 0]],
                    [[1, 1, 1], [-4, -4, -4], [1, 1, 1]],
                    [[0, 0, 0], [1, 1, 1], [0, 0, 0]]])

    y, err = con.convolution(w, x, Reversal.REV, ConvolutionType.Other, 3, 0, 0)

    show_data(y, ImageDataType.RGB)

    # 滤波器 w（提取边缘特征）
    w = np.asarray([[[0, 0, 0], [1, 1, 1], [1, 1, 1]],
                    [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
                    [[-1, -1, -1], [-1, -1, -1], [0, 0, 0]]])

    y, err = con.convolution(w, x, Reversal.REV, ConvolutionType.Other, 3, 0, 0)

    show_data(y, ImageDataType.GRAY)


"""
功能：测试 max pooling，2维
参数：NULL 
返回值：NULL
"""


def test_max_pooling_2d():
    w, x = _create_w_x_2d()

    con = MaxPooling()

    y, err = con.convolution(w, x, Reversal.REV, ConvolutionType.Other, 2, 0, 0)

    print("\n 3维，max pooling\n")

    print(y)


"""
功能：测试 max pooling，3维
参数：NULL 
返回值：NULL
"""


def test_max_pooling_3d():
    w, x = _create_w_x_3d()

    con = MaxPooling()

    y, err = con.convolution(w, x, Reversal.REV, ConvolutionType.Other, 2, 0, 0)

    print("\n 3维，max pooling\n")

    print(y)


"""
功能：测试 mean pooling，2维
参数：NULL 
返回值：NULL
"""


def test_mean_pooling_2d():
    w, x = _create_w_x_2d()

    con = MeanPooling()

    y, err = con.convolution(w, x, Reversal.REV, ConvolutionType.Other, 2, 0, 0)

    print("\n 2维，mean pooling\n")

    print(y)


"""
功能：测试 mean pooling，3维
参数：NULL 
返回值：NULL
"""


def test_mean_pooling_3d():
    w, x = _create_w_x_3d()

    con = MeanPooling()

    y, err = con.convolution(w, x, Reversal.REV, ConvolutionType.Other, 2, 0, 0)

    print("\n 3维，mean pooling\n")

    print(y)


"""
功能：构建 w, x，2维
参数：NULL 
返回值：NULL
"""


def _create_w_x_2d():
    # 输入信息 x
    x = np.asarray([[1, 1, 2, 0],
                    [0, 1, 1, 2],
                    [0, 0, 1, 3],
                    [0, 0, 1, 1]])

    # 卷积核
    w = np.zeros([2, 2])

    return w, x


"""
功能：构建 w, x，3维
参数：NULL 
返回值：NULL
"""


def _create_w_x_3d():
    # 输入信息 x
    x = np.asarray([[[1], [1], [2], [0]],
                    [[0], [1], [1], [2]],
                    [[0], [0], [1], [3]],
                    [[0], [0], [1], [1]]])

    # 卷积核
    w = np.zeros([2, 2, 1])

    return w, x
