"""
Function：感知器测试
Author：lzb
Date：2021.01.02
"""

import numpy as np

from gl import draw
from sample import straight_line_sample
from fnn.test_fnn import NNTest
from pt.perceptron import Perceptron

# 感知器 title
PERCEPTRON_TITLE = "Perceptron"

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

    # 感知器对象
    nn = Perceptron()

    # 每一层网络的神经元个数（感知器只有1层，只有1个神经元）
    neuron_count_list = [1]

    # 最大循环训练次数
    loop_max = 10

    # 学习效率
    rate = 0.15

    # 预测样本参数
    predict_sample_count = 500

    # 预测样本，输入，每个向量元素的最大值
    predict_sx_max = list()
    predict_sx_max.append(2000)
    predict_sx_max.append(5000)

    # 是否画训练样本
    draw_train_sample_flag = draw.ShowFlag.SHOW

    # 是否画预测样本
    draw_predict_sample_flag = draw.ShowFlag.SHOW

    # 是否画预测结果
    draw_predict_result_flag = draw.ShowFlag.SHOW

    test = NNTest()

    test.test(train_sample_count, train_sx_max, sample, neuron_count_list, loop_max, rate,
              predict_sample_count, predict_sx_max, nn, PERCEPTRON_TITLE,
              draw_train_sample_flag, draw_predict_sample_flag, draw_predict_result_flag)


"""
功能：测试 ax + b 分割，不经过训练，直接赋值参数
参数：NULL    
返回值：NULL
"""


def test_straight_line_without_train():
    # 神经网络对象
    nn = Perceptron()

    # 训练样本个数
    predict_sample_count = 500

    # 训练样本，输入，每个向量元素的最大值
    predict_sx_max = list()
    predict_sx_max.append(2000)
    predict_sx_max.append(5000)

    # 神经网络输入/输出，向量维度
    sx_dim = 2
    sy_dim = 1

    # 每一层神经元的数量(Neuron Count)
    neuron_count_list = [1]

    # 每一层 w、B 参数，w 是个 matrix，b 是个 vector（数据类型也是一个 matrix）
    W = list()
    B = list()

    # 第1层（只有1层）
    w0 = np.asarray([[-402.07831651, 209.5955333]])
    b0 = np.asarray([[25.94985019]])

    W.append(w0)
    B.append(b0)

    # ax + b 分割的样本对象
    sample = straight_line_sample.StraightLineSample()

    # 是否画预测样本
    draw_predict_sample_flag = draw.ShowFlag.NO_SHOW

    # 是否画预测结果
    draw_predict_result_flag = draw.ShowFlag.SHOW

    test = NNTest()

    test.test_stub(predict_sample_count, predict_sx_max, neuron_count_list, W, B, sample, nn, PERCEPTRON_TITLE,
                   draw_predict_sample_flag, draw_predict_result_flag,
                   sx_dim, sy_dim)


