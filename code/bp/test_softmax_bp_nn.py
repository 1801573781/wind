"""
Function：softmax 测试
Author：lzb
Date：2021.02.02
"""

import os

import numpy as np

from activation.last_hop_activation import LastHopActivation, SoftMaxLHA
from activation.normal_activation import Sigmoid, ReLU
from loss.loss import MSELoss, CrossEntropyLoss

from bp.bp_nn_recitation import Recitation
from bp import bp_nn, bp_nn_ex
from sample.image_softmax_sample import ImageSoftMaxSample
from sample.rnn_sample import RNNSanmple


def test_softmax():
    # 1. 构建神经网络对象

    # 激活函数
    # activation = Sigmoid()
    activation = ReLU(20)

    # 最后一跳激活函数
    last_hop_activation = SoftMaxLHA()

    # 损失函数
    loss = CrossEntropyLoss()

    # 神经网络
    # nn = bp_nn.BPFNN(activation, last_hop_activation, loss)
    nn = bp_nn_ex.BPFnnEx(activation, last_hop_activation, loss)

    # 2. 构建训练样本

    # 训练样本对象
    sample = ImageSoftMaxSample()

    # 训练样本输入，向量维度
    sx_dim = 400  # 20 * 20 的图像， 400维向量

    # 训练样本输出，向量维度
    sy_dim = 10  # one-hot, 10维向量

    # 创建训练样本，输入/输出
    train_image_root_path = "./../picture/number_softmax_train"
    train_image_root_path = os.path.abspath(train_image_root_path)

    sample.create_sample(train_image_root_path)
    # sample.create_sample_ex(100)

    # train_sx_list = sample.get_sx_list()
    # train_sy_list = sample.get_sy_list()

    train_sx_group = sample.get_sx_group()
    train_sy_group = sample.get_sy_group()

    # 3. 训练

    # 每一层网络的神经元个数
    neuron_count_list = [10, 10]

    # 最大循环训练次数
    loop_max = 30

    # 学习效率
    rate = 0.1

    # 训练
    # nn.train(train_sx_list, train_sy_list, loop_max, neuron_count_list, rate)
    nn.train(train_sx_group, train_sy_group, loop_max, neuron_count_list, rate)

    # 4. 测试

    # 4.1 创建测试样本
    test_image_root_path = "./../picture/number_softmax_test"
    test_image_root_path = os.path.abspath(test_image_root_path)

    sample.create_sample(test_image_root_path, confuse=False)
    # sample.create_sample_ex(2)

    test_sx_list = sample.get_sx_list()
    test_sy_list = sample.get_sy_list()

    py_list = nn.predict(test_sx_list, test_sy_list)

    print("\n")
    print("py:\n")

    count = len(py_list)

    for i in range(0, count):
        number = _get_max_index(test_sy_list[i])

        print("\n")
        print("index = %d, number = %d" % (i, number))
        print(py_list[i])


''''''


def _get_max_index(y):
    """
    获取 y 中概率最大的那个元素的索引（并且该值大于等于0.9）
    :param y: 或者是 sy（训练样本输出），或者是 py（预测结果输出）
    :return: y 中概率最大的那个元素的索引
    """

    row = y.shape[0]

    for r in range(0, row):
        # 如果小于 0.1 则认为是0
        if y[r][0] >= 0.9:
            return r

    # 如果没有大于等于0.9的，则 return -1
    return -1


''''''


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
    # nn = bp_nn.BPFNN(activation, last_hop_activation, loss)
    nn = Recitation(activation, last_hop_activation, loss)

    # 2. 构建训练样本

    # 训练样本对象
    sample = RNNSanmple()

    sample.create_sample()

    train_sx_group = sample.get_sx_group()
    train_sy_group = sample.get_sy_group()

    # 3. 训练

    # 每一层网络的神经元个数
    neuron_count_list = [10, 21]

    # 最大循环训练次数
    loop_max = 50

    # 学习效率
    rate = 0.1

    # 训练
    nn.train(train_sx_group, train_sy_group, loop_max, neuron_count_list, rate)

    # 4. 测试

    # 4.1 创建测试样本
    ch = "白"
    test_sx = sample.create_test_sample(ch)

    # 测试
    py_list = list()
    nn.predict_recurrent(test_sx, py_list)

    # 将测试样本放在首位，这样就组成了一首完整的诗
    py_list.insert(0, ch)

    print("\n")
    print("py_list:\n")

    print(py_list)
