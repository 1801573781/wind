"""
Function：rnn 测试
Author：lzb
Date：2021.02.15
"""

import os

from activation.last_hop_activation import LastHopActivation, SoftMaxLHA
from activation.normal_activation import Sigmoid, ReLU

from loss.loss import MSELoss, CrossEntropyLoss

from bp import bp_nn
from rnn import recurrent_nn
from sample.image_softmax_sample import ImageSoftMaxSample
from sample.rnn_sample import RNNSanmple


def test_poem():
    # 1. 构建神经网络对象

    # 激活函数
    # activation = Sigmoid()
    activation = ReLU(20)

    # 最后一跳激活函数
    last_hop_activation = SoftMaxLHA()

    # 损失函数
    loss = CrossEntropyLoss()

    # 神经网络
    nn = recurrent_nn.RecurrentNN(activation, last_hop_activation, loss)

    # 2. 构建训练样本

    # 训练样本对象
    sample = RNNSanmple()

    sample.create_sample()

    train_sx_list = sample.get_sx_list()
    train_sy_list = sample.get_sy_list()

    # 3. 训练

    # 每一层网络的神经元个数
    neuron_count_list = [10, 21]

    # 最大循环训练次数
    loop_max = 50

    # 学习效率
    rate = 0.1

    # 训练
    nn.train(train_sx_list, train_sy_list, loop_max, neuron_count_list, rate)

    # 4. 测试

    # 4.1 创建测试样本
    test_sx = sample.create_test_sample("床")

    # 测试
    py_list = list()
    nn.predict(test_sx, py_list)

    print("\n")
    print("py:\n")

    print(py_list)

