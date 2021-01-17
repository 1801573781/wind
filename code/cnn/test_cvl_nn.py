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
from my_image import my_image
from my_image.my_image import show_file, gray_file, show_data, ImageDataType, get_data
from activation.active import Sigmoid
from activation.active import ReLU

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
功能：测试卷积神经网络
参数：NULL 
返回值：NULL
"""


def test_cvl_nn():
    # 图像数据
    file_name = "../picture/number/0.bmp"
    # file_name = "../picture/base_test/dog1.png"

    show_file(file_name)

    data, image_data_type, err = get_data(file_name)

    # 归一化
    data = my_image.normalize(data, my_image.NormalizationType.NORMAL)

    # 训练输入样本
    sx_list = list()
    sx_list.append(data)

    # 训练输出样本
    sy_list = list()
    sy_list.append(data)  # 先随便赋个值，暂时还用不到这个数据

    # 卷积神经网络的基本参数

    # 每一层网络的神经元个数(这个参数，对于卷积网络而言，没意义)
    neuron_count_list = None

    # 最大循环训练次数
    loop_max = 1

    # 学习效率
    rate = 0.1

    # 卷积核数组大小
    w_shape_list = list()

    w1_shape = [5, 5, 3]
    w2_shape = [3, 3, 3]

    w_shape_list.append(w1_shape)
    w_shape_list.append(w2_shape)

    # 激活函数
    # activation = Sigmoid()
    activation = ReLU()

    # 构建卷积神经网络对象
    cnn = CVLNeuralNetwork()

    # 训练
    cnn.train(sx_list, sy_list, loop_max, neuron_count_list, rate, activation, w_shape_list)

    # 预测
    py_list = cnn.predict(sx_list, sy_list)

    # 将预测结果显示如初
    py = py_list[0]

    py = my_image.normalize(py, my_image.NormalizationType.REV_NORMAL)

    py = my_image.array_3_2(py)

    show_data(py, ImageDataType.GRAY)
