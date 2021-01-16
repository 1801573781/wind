"""
Function：卷积神经测试
Author：lzb
Date：2021.01.14

特别说明：来到了卷积网络，这几天编码的效率有点低
"""

import numpy as np
import os

from cnn.max_pooling import MaxPooling
from cnn.mean_pooling import MeanPooling
from cnn.convolution import Convolution, Reversal, ConvolutionType
from cnn.cvl_nn import CVLNeuralNetwork
from my_image.my_image import show_file, gray_file, show_data, ImageDataType, get_data

"""
功能：测试卷积神经网络
参数：NULL 
返回值：NULL
"""


def test_cvl_nn():
    # 图像数据
    file_name = "../picture/base_test/dog1.bmp"

    show_file(file_name)

    data, image_data_type, err = get_data(file_name)

    # 训练样本
    sx_list = list()
    sx_list.append(data)

    # 卷积神经网络的基本参数

    # 每一层网络的神经元个数(这个参数，对于卷积网络而言，没意义)
    neuron_count_list = 0

    # 最大循环训练次数
    loop_max = 2000

    # 学习效率
    rate = 0.1

    # 卷积核
    W = list()

    w1 = np.zeros([3, 3, 1])
    W.append(w1)

    w2 = np.zeros([3, 3, 1])
    W.append(w2)
