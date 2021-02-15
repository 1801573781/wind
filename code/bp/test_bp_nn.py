"""
Function：神经网络测试
Author：lzb
Date：2021.01.01
"""

import numpy as np
import os

from activation import normal_activation
from gl import draw
from bp import bp_nn
from sample import sin_sample
from sample import straight_line_sample
from sample import two_line_sample
from nn.test_nn import NNTest

# BP 神经网络 title
BP_NN_TITLE = "BP Neural Network"


"""
功能：测试 ax + b 分割
参数：NULL    
返回值：NULL
"""


def test_straight_line():
    # 训练样本个数
    train_sample_count = 1000

    # 训练样本，输入，每个向量元素的最大值
    train_sx_max = list()
    train_sx_max.append(200)
    train_sx_max.append(500)

    # ax + b 分割的样本对象
    sample = straight_line_sample.StraightLineSample()

    # 激活函数对象
    activation = normal_activation.Sigmoid()

    # 神经网络对象
    nn = bp_nn.BPFNN(activation)

    # 每一层网络的神经元个数
    neuron_count_list = [2, 1]

    # 最大循环训练次数
    loop_max = 10

    # 学习效率
    rate = 0.1

    # 预测样本参数
    predict_sample_count = 500

    # 预测样本，输入，每个向量元素的最大值
    predict_sx_max = list()
    predict_sx_max.append(2000)
    predict_sx_max.append(5000)

    # 是否画训练样本
    draw_train_sample_flag = draw.ShowFlag.NO_SHOW

    # 是否画预测样本
    draw_predict_sample_flag = draw.ShowFlag.NO_SHOW

    # 是否画预测结果
    draw_predict_result_flag = draw.ShowFlag.SHOW

    test = NNTest()

    test.test(train_sample_count, train_sx_max, sample, neuron_count_list, loop_max, rate,
              predict_sample_count, predict_sx_max, nn, BP_NN_TITLE,
              draw_train_sample_flag, draw_predict_sample_flag, draw_predict_result_flag)


"""
功能：测试 两根直线 分割
参数：NULL    
返回值：NULL
"""


def test_two_line():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 训练样本个数
    train_sample_count = 1000

    # 训练样本，输入，每个向量元素的最大值
    train_sx_max = list()
    train_sx_max.append(1)
    train_sx_max.append(1)

    # 两根直线 分割的样本对象
    sample = two_line_sample.TwoLineSample()

    # 激活函数对象
    activation = normal_activation.Sigmoid()

    # 神经网络对象
    nn = bp_nn.BPFNN(activation)

    # 每一层网络的神经元个数
    neuron_count_list = [2, 1]

    # 最大循环训练次数
    loop_max = 1000

    # 学习效率
    rate = 0.1

    # 预测样本参数
    predict_sample_count = 500

    # 预测样本，输入，每个向量元素的最大值
    predict_sx_max = list()
    predict_sx_max.append(1)
    predict_sx_max.append(1)

    # 是否画训练样本
    draw_train_sample_flag = draw.ShowFlag.SHOW

    # 是否画预测样本
    draw_predict_sample_flag = draw.ShowFlag.SHOW

    # 是否画预测结果
    draw_predict_result_flag = draw.ShowFlag.SHOW

    test = NNTest()

    test.test(train_sample_count, train_sx_max, sample, neuron_count_list, loop_max, rate,
              predict_sample_count, predict_sx_max, nn, BP_NN_TITLE,
              draw_train_sample_flag, draw_predict_sample_flag, draw_predict_result_flag)


"""
功能：测试 sin(x) 分割
参数：NULL    
返回值：NULL
"""


def test_sin():
    # 训练样本个数
    train_sample_count = 1000

    # 训练样本，输入，每个向量元素的最大值
    train_sx_max = list()
    train_sx_max.append(1)
    train_sx_max.append(1)

    # sin(x) 分割的样本对象
    sample = sin_sample.SinSample()

    # 激活函数对象
    activation = normal_activation.Sigmoid()

    # 神经网络对象
    nn = bp_nn.BPFNN(activation)

    # 每一层网络的神经元个数
    neuron_count_list = [5, 1]

    # 最大循环训练次数
    loop_max = 2000

    # 学习效率
    rate = 0.1

    # 预测样本参数
    predict_sample_count = 1000

    # 预测样本，输入，每个向量元素的最大值
    predict_sx_max = list()
    predict_sx_max.append(1)
    predict_sx_max.append(1)

    # 是否画训练样本
    draw_train_sample_flag = draw.ShowFlag.SHOW

    # 是否画预测样本
    draw_predict_sample_flag = draw.ShowFlag.SHOW

    # 是否画预测结果
    draw_predict_result_flag = draw.ShowFlag.SHOW

    test = NNTest()

    test.test(train_sample_count, train_sx_max, sample, neuron_count_list, loop_max, rate,
              predict_sample_count, predict_sx_max, nn, BP_NN_TITLE,
              draw_train_sample_flag, draw_predict_sample_flag, draw_predict_result_flag)


"""
功能：测试 ax + b 分割，不经过训练，直接赋值参数
参数：NULL    
返回值：NULL
"""


def test_straight_line_without_train():
    # 激活函数对象
    activation = normal_activation.Sigmoid()

    # 神经网络对象
    nn = bp_nn.BPFNN(activation)

    # 预测样本个数
    predict_sample_count = 500

    # 预测样本，输入，每个向量元素的最大值
    predict_sx_max = list()
    predict_sx_max.append(2000)
    predict_sx_max.append(5000)

    # 神经网络输入/输出，向量维度
    sx_dim = 2
    sy_dim = 1

    # 每一层神经元的数量(Neuron Count)
    neuron_count_list = [2, 1]

    # 每一层 w、B 参数，w 是个 matrix，b 是个 vector（数据类型也是一个 matrix）
    W = list()
    B = list()

    # 第1层
    w0 = np.asarray([[-18.82131354, 9.6738681],
                     [-8.15464853, 7.34378254]])

    b0 = np.asarray([[0.07273821],
                     [0.22575042]])

    W.append(w0)
    B.append(b0)

    # 第2层
    w1 = np.asarray([[8.96780964, 0.42862565]])
    b1 = np.asarray([[-3.94002764]])

    W.append(w1)
    B.append(b1)

    # ax + b 分割的样本对象
    sample = straight_line_sample.StraightLineSample()

    # 是否画预测样本
    draw_predict_sample_flag = draw.ShowFlag.NO_SHOW

    # 是否画预测结果
    draw_predict_result_flag = draw.ShowFlag.SHOW

    test = NNTest()

    test.test_stub(predict_sample_count, predict_sx_max, neuron_count_list, W, B, sample, nn, BP_NN_TITLE,
                   draw_predict_sample_flag, draw_predict_result_flag,
                   sx_dim, sy_dim)


"""
功能：测试 两根直线 分割，不经过训练，直接赋值参数
参数：NULL    
返回值：NULL
"""


def test_two_line_without_train():
    # 激活函数对象
    activation = normal_activation.Sigmoid()

    # 神经网络对象
    nn = bp_nn.BPFNN(activation)

    # 预测样本个数
    predict_sample_count = 1000

    # 预测样本，输入，每个向量元素的最大值
    predict_sx_max = list()
    predict_sx_max.append(1)
    predict_sx_max.append(1)

    # 神经网络输入/输出，向量维度
    sx_dim = 2
    sy_dim = 1

    # 每一层神经元的数量(Neuron Count)
    neuron_count_list = [2, 1]

    # 每一层 w、B 参数，w 是个 matrix，b 是个 vector（数据类型也是一个 matrix）
    W = list()
    B = list()

    # 第1层
    w0 = np.asarray([[15.3875926, -15.33421447],
                     [-14.19520774, 14.33819683]])

    b0 = np.asarray([[-7.82673954],
                     [-6.9671098]])

    W.append(w0)
    B.append(b0)

    # 第2层
    w1 = np.asarray([[20.86723432, 19.43161973]])
    b1 = np.asarray([[-9.78812464]])

    W.append(w1)
    B.append(b1)

    # 两根直线 分割的样本对象
    sample = two_line_sample.TwoLineSample()

    # 是否画预测样本
    draw_predict_sample_flag = draw.ShowFlag.NO_SHOW

    # 是否画预测结果
    draw_predict_result_flag = draw.ShowFlag.SHOW

    test = NNTest()

    test.test_stub(predict_sample_count, predict_sx_max, neuron_count_list, W, B, sample, nn, BP_NN_TITLE,
                   draw_predict_sample_flag, draw_predict_result_flag,
                   sx_dim, sy_dim)


"""
功能：测试 两根直线 分割，不经过训练，直接赋值参数
参数：NULL    
返回值：NULL
"""


def test_sin_without_train():
    # 激活函数对象
    activation = normal_activation.Sigmoid()

    # 神经网络对象
    nn = bp_nn.BPFNN(activation)

    # 预测样本个数
    predict_sample_count = 1000

    # 预测样本，输入，每个向量元素的最大值
    predict_sx_max = list()
    predict_sx_max.append(1)
    predict_sx_max.append(1)

    # 神经网络输入/输出，向量维度
    sx_dim = 2
    sy_dim = 1

    # 每一层神经元的数量(Neuron Count)
    neuron_count_list = [5, 1]

    # 每一层 w、B 参数，w 是个 matrix，b 是个 vector（数据类型也是一个 matrix）
    W = list()
    B = list()

    # 第1层

    w0 = np.asarray([[-20.69514968, - 3.52723041],
                     [-11.69201431, 2.12957366],
                     [20.53242169, 3.31010044],
                     [-18.77519677, 3.54389238],
                     [-9.44057119, 2.41633713]])

    b0 = np.asarray([[-9.81608468],
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

    # sin(x) 分割的样本对象
    sample = sin_sample.SinSample()

    # 是否画预测样本
    draw_predict_sample_flag = draw.ShowFlag.NO_SHOW

    # 是否画预测结果
    draw_predict_result_flag = draw.ShowFlag.SHOW

    test = NNTest()

    test.test_stub(predict_sample_count, predict_sx_max, neuron_count_list, W, B, sample, nn, BP_NN_TITLE,
                   draw_predict_sample_flag, draw_predict_result_flag,
                   sx_dim, sy_dim)
