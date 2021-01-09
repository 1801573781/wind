"""
Function：卷积测试
Author：lzb
Date：2021.01.09
"""

import numpy as np
import os

from cnn.convolution import Convolution
from cnn.convolution import Reversal
from cnn.convolution import ConvolutionType
from my_image.image import show_file, gray_file, show_data, ImageDataType

"""
功能：测试卷积网络-1
参数：NULL 
返回值：NULL
"""


def test1():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # 图像数据（灰度）
    file_name = "./../my_image/dog1.bmp"

    show_file(file_name)

    gray, err = gray_file(file_name)
    show_data(gray, ImageDataType.GRAY)

    # 卷积
    con = Convolution()

    # 输入信息 x
    x = gray

    # 滤波器 w（平滑去噪）
    w = np.asarray([[1/16, 1/8, 1/16],
                    [1/8, 1/4, 1/8],
                    [1/16, 1/8, 1/16]])

    y, err = con.convolution_d2(w, x, Reversal.REV, ConvolutionType.Narrow)

    show_data(y, ImageDataType.GRAY)

    # 滤波器 w（提取边缘特征）
    w = np.asarray([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])

    y, err = con.convolution_d2(w, x, Reversal.REV, ConvolutionType.Narrow)

    show_data(y, ImageDataType.GRAY)

    # 滤波器 w（提取边缘特征）
    w = np.asarray([[0, 1, 1],
                    [-1, 0, 1],
                    [-1, -1, 0]])

    y, err = con.convolution_d2(w, x, Reversal.REV, ConvolutionType.Narrow)

    show_data(y, ImageDataType.GRAY)
