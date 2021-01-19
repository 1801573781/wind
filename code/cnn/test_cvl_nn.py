"""
Function：卷积神经测试
Author：lzb
Date：2021.01.14

特别说明：来到了卷积网络，这几天编码的效率有点低
"""

import numpy as np
import numpy.matlib
import os

from cnn.max_pooling import MaxPooling
from cnn.mean_pooling import MeanPooling
from cnn.convolution import Convolution, Reversal, ConvolutionType
from cnn.cvl_nn import CVLNeuralNetwork
from gl.common_enum import ArrayDim
from gl.common_function import *
from my_image import my_image
from my_image.my_image import show_file, gray_file, show_data, ImageDataType, get_data
from activation.active import Sigmoid
from activation.active import ReLU

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test():
    # arr1
    arr1 = np.random.random(3)

    print("\narr1:\n")

    print(array_string(arr1))

    # arr2
    arr2 = np.matlib.rand(3, 3)

    print("\narr2:\n")

    print(array_string(arr2))

    # arr3
    arr3 = rand_array_3(3, 3, 3)

    print("\narr3:\n")

    print(array_string(arr3))



"""
功能：测试卷积神经网络
参数：NULL 
返回值：NULL
"""


def test_cvl_nn():
    # 1. 构建训练输入样本

    # 图像文件名
    file_name = "../picture/number/9_sx.bmp"

    # 取灰度值
    data, err = gray_file(file_name, ArrayDim.THREE)

    # 将图像数据中的0转换为极小值
    my_image.array_0_tiny(data)

    # 显示灰度图像
    gray = my_image.array_3_2(data)
    show_data(gray, ImageDataType.GRAY)

    # 归一化
    sx = my_image.normalize(data, my_image.NormalizationType.NORMAL)

    # 显示归一化灰度图像
    gray = my_image.array_3_2(sx)
    show_data(gray, ImageDataType.GRAY)

    # 训练输入样本
    sx_list = list()
    sx_list.append(sx)

    # 2. 构建训练输出样本

    # 图像数据
    file_name = "../picture/number/9_sy_2.bmp"

    # 取灰度值
    data, err = gray_file(file_name, ArrayDim.THREE)

    # 将图像数据中的0转换为极小值
    my_image.array_0_tiny(data)

    # 显示灰度图像
    gray = my_image.array_3_2(data)
    show_data(gray, ImageDataType.GRAY)

    # 归一化
    sy = my_image.normalize(data, my_image.NormalizationType.NORMAL)

    # 显示归一化灰度图像
    gray = my_image.array_3_2(sy)
    show_data(gray, ImageDataType.GRAY)

    # 训练输出样本
    sy_list = list()
    sy_list.append(sy)

    # 3. 卷积神经网络的基本参数

    # 每一层网络的神经元个数(这个参数，对于卷积网络而言，没意义)
    neuron_count_list = None

    # 最大循环训练次数
    loop_max = 10000

    # 学习效率
    rate = 0.01

    # 卷积核数组大小
    w_shape_list = list()

    w1_shape = [3, 3, 1]
    w2_shape = [3, 3, 1]

    w_shape_list.append(w1_shape)
    w_shape_list.append(w2_shape)

    # 激活函数
    activation = Sigmoid()
    # activation = ReLU()

    # 构建卷积对象
    cvl = Convolution()

    # 构建卷积神经网络对象
    cnn = CVLNeuralNetwork(cvl)

    # 训练
    cnn.train(sx_list, sy_list, loop_max, neuron_count_list, rate, activation, w_shape_list)

    # 预测
    py_list = cnn.predict(sx_list, sy_list)

    # 将预测结果显示如初
    py = py_list[0]

    py = my_image.normalize(py, my_image.NormalizationType.REV_NORMAL)

    py = my_image.array_3_2(py)

    show_data(py, ImageDataType.GRAY)



"""
功能：测试卷积神经网络，不经过训练，直接赋值参数
参数：NULL 
返回值：NULL
"""


def test_cvl_nn_without_train():
    # 1. 构建训练输入样本

    # 图像文件名
    file_name = "../picture/number/9_sx.bmp"

    # 取灰度值
    data, err = gray_file(file_name, ArrayDim.THREE)

    # 将图像数据中的0转换为极小值
    my_image.array_0_tiny(data)

    # 显示灰度图像
    gray = my_image.array_3_2(data)
    show_data(gray, ImageDataType.GRAY)

    # 归一化
    sx = my_image.normalize(data, my_image.NormalizationType.NORMAL)

    # 显示归一化灰度图像
    gray = my_image.array_3_2(sx)
    show_data(gray, ImageDataType.GRAY)

    # 训练输入样本
    sx_list = list()
    sx_list.append(sx)

    # 2. 构建训练输出样本

    # 图像数据
    file_name = "../picture/number/9_sy_2.bmp"

    # 取灰度值
    data, err = gray_file(file_name, ArrayDim.THREE)

    # 将图像数据中的0转换为极小值
    my_image.array_0_tiny(data)

    # 显示灰度图像
    gray = my_image.array_3_2(data)
    show_data(gray, ImageDataType.GRAY)

    # 归一化
    sy = my_image.normalize(data, my_image.NormalizationType.NORMAL)

    # 显示归一化灰度图像
    gray = my_image.array_3_2(sy)
    show_data(gray, ImageDataType.GRAY)

    # 训练输出样本
    sy_list = list()
    sy_list.append(sy)

    # 3. 卷积神经网络的基本参数

    ''' W, B 参数 begin '''

    # 每一层 w、B 参数，w 是个 matrix，b 是个 vector（数据类型也是一个 matrix）
    W = list()
    B = list()

    # 第1层

    w0 = np.asarray([[[56.30099503], [93.56124584], [79.86847495]],
                     [[78.1288688], [109.1631134], [67.61090819]],
                     [[98.6293555], [94.77138308], [46.62941498]]])

    b0 = np.asarray([[4.81443329e-01],
                     [-11.76723006],
                     [-9.91804491],
                     [-0.22741755],
                     [9.6019504]])

    W.append(w0)
    B.append(b0)

    # 第2层

    w1 = np.asarray([[-16.96090009, 14.45759092, 13.19833084, 17.00121887, 10.07422065]])

    b1 = np.asarray([[-17.35104879]])

    W.append(w1)
    B.append(b1)

    ''' W, B 参数 end '''

    # 激活函数
    activation = Sigmoid()
    # activation = ReLU()

    # 构建卷积对象
    cvl = Convolution()

    # 构建卷积神经网络对象
    cnn = CVLNeuralNetwork(cvl)

    # 训练
    cnn.train(sx_list, sy_list, loop_max, neuron_count_list, rate, activation, w_shape_list)

    # 预测
    py_list = cnn.predict(sx_list, sy_list)

    # 将预测结果显示如初
    py = py_list[0]

    py = my_image.normalize(py, my_image.NormalizationType.REV_NORMAL)

    py = my_image.array_3_2(py)

    show_data(py, ImageDataType.GRAY)