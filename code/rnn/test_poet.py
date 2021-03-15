"""
Function：测试写诗机
Author：lzb
Date：2021.03.05
"""


from gl.common_function import unserialize_train_para

import os

from activation.last_hop_activation import SoftMaxLHA
from activation.normal_activation import LeakyReLU
from loss.loss import CrossEntropyLoss
from rnn.rnn_poet import Poet
from sample.poem_sample import PoemSample


def test_poet():
    """
    测试 rnn_poet（写诗机的第一步）
    :return: NULL
    """

    # 1. 构建神经网络对象

    # 激活函数
    # activation = Sigmoid()
    # activation = ReLU(20)
    activation = LeakyReLU(20)

    # 最后一跳激活函数
    last_hop_activation = SoftMaxLHA()

    # 损失函数
    loss = CrossEntropyLoss()

    # 神经网络
    nn = Poet(activation, last_hop_activation, loss)

    # 2. 构建训练样本

    # 训练样本对象
    sample = PoemSample()

    sample.create_sample_group()

    train_sx_group = sample.get_sx_group()
    train_sy_group = sample.get_sy_group()

    # 3. 训练

    # 每一层网络的神经元个数
    one_hot_dim = sample.get_one_hot_dim()
    neuron_count_list = [10, one_hot_dim]

    # 最大循环训练次数
    loop_max = 10000

    # 学习效率
    rate = 0.1

    # 训练
    init_from_unserialization = True
    alpha_para = 0.1
    nn.train(train_sx_group, train_sy_group, loop_max, neuron_count_list, rate, init_from_unserialization, alpha_para)

    # 4. 测试

    # 创建测试样本
    # 诗的起始字符
    ch = "锄"
    # ch = "小"
    test_sx = sample.create_test_sample(ch)

    # 测试
    py_list = list()
    nn.predict_recurrent(test_sx, py_list)

    # 将测试样本放在首位，这样就组成了一首完整的诗
    py_list.insert(0, ch)

    print("\n")
    print("py:\n")

    print(py_list)


''''''


def test_poet_without_train():
    """
    测试 rnn，不经过训练，直接赋值参数
    :return: NULL
    """

    # 激活函数
    # activation = Sigmoid()
    # activation = ReLU(20)
    activation = LeakyReLU(20)

    # 最后一跳激活函数
    last_hop_activation = SoftMaxLHA()

    # 损失函数
    loss = CrossEntropyLoss()

    # 神经网络
    nn = Poet(activation, last_hop_activation, loss)

    # 测试样本对象
    sample = PoemSample()

    # 每一层网络的神经元个数
    one_hot_dim = sample.get_one_hot_dim()
    neuron_count_list = [10, 10, 10, one_hot_dim]

    # 反序列化每一层 w, b, u 参数
    file_path = os.path.dirname(__file__) + "/../gl/train_para/"
    layer_count = len(neuron_count_list)
    w_layer, b_layer, u_layer = unserialize_train_para(file_path, layer_count, u_flag=True)

    # nn 设置参数
    nn.stub_set_para(neuron_count_list, w_layer, b_layer, u_layer)

    # 创建测试样本
    # ch = "小"
    ch = "白"
    test_sx = sample.create_test_sample(ch)

    # 测试
    py_list = list()
    nn.predict_recurrent(test_sx, py_list)

    # 将测试样本放在首位，这样就组成了一首完整的诗
    py_list.insert(0, ch)

    print("\n")
    print("py:\n")

    print(py_list)
