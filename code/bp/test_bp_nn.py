"""
Function：神经网络测试
Author：lzb
Date：2021.01.01
"""


import numpy as np

from gl import draw
from bp import bp_nn
from sample import sin_sample
from sample import straight_line_sample
from sample import two_line_sample
from nn.test_nn import NNTest

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

    # 神经网络对象
    nn = bp_nn.BPNeuralNetwork()

    # title
    title = "BP 神经网络"

    # 每一层网络的神经元个数
    neuron_count_list = [5, 1]

    # 最大循环训练次数
    loop_max = 1000

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
              predict_sample_count, predict_sx_max, nn, title,
              draw_train_sample_flag, draw_predict_sample_flag, draw_predict_result_flag)


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

    # 神经网络对象
    nn = bp_nn.BPNeuralNetwork()

    # title
    title = "BP 神经网络"

    # 每一层网络的神经元个数
    neuron_count_list = [2, 1]

    # 最大循环训练次数
    loop_max = 10

    # 学习效率
    rate = 0.20

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
              predict_sample_count, predict_sx_max, nn, title,
              draw_train_sample_flag, draw_predict_sample_flag, draw_predict_result_flag)


"""
功能：测试 两根直线 分割
参数：NULL    
返回值：NULL
"""


def test_two_line():
    # 训练样本个数
    train_sample_count = 1000

    # 训练样本，输入，每个向量元素的最大值
    train_sx_max = list()
    train_sx_max.append(1)
    train_sx_max.append(1)

    # 两根直线 分割的样本对象
    sample = two_line_sample.TwoLineSample()

    # 神经网络对象
    nn = bp_nn.BPNeuralNetwork()

    # title
    title = "BP 神经网络"

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
              predict_sample_count, predict_sx_max, nn, title,
              draw_train_sample_flag, draw_predict_sample_flag, draw_predict_result_flag)


"""
功能：测试 两根直线 分割，不经过训练，直接赋值参数
参数：NULL    
返回值：NULL
"""


def test_two_line_without_train():
    # 神经网络对象
    nn = bp_nn.BPNeuralNetwork()

    # title
    title = "BP 神经网络(未训练)"

    # 神经网络输入，向量维度
    sx_dim = 2

    # 神经网络层数
    layer_count = 2

    # test github

    # 每一层神经元的数量(Neuron Count)
    neuron_count_list = [2, 1]

    # 每一层 w 参数，w 是个 matrix
    W = list()

    w0 = np.asarray([[15.3875926, -15.33421447], [-14.19520774, 14.33819683]])
    w1 = np.asarray([[20.86723432, 19.43161973]])

    W.append(w0)
    W.append(w1)

    # 每一层 b 参数，b 是个 vector
    B = list()

    b0 = np.asarray([[-7.82673954], [-6.9671098]])
    b1 = np.asarray([[-9.78812464]])

    B.append(b0)
    B.append(b1)

    # 两根直线 分割的样本对象
    sample = two_line_sample.TwoLineSample()

    # 预测样本，输入，每个向量元素的最大值
    predict_sx_max = list()
    predict_sx_max.append(20)
    predict_sx_max.append(40)

    # 是否画预测样本
    draw_predict_sample_flag = draw.ShowFlag.NO_SHOW

    # 是否画预测结果
    draw_predict_result_flag = draw.ShowFlag.SHOW

    test = NNTest()

    test.test_stub(sx_dim, layer_count, neuron_count_list, W, B, sample, nn, title,
                   draw_predict_sample_flag, draw_predict_result_flag)


