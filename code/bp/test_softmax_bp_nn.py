"""
Function：softmax 测试
Author：lzb
Date：2021.02.02
"""


import os

from activation.last_hop_activation import LastHopActivation, SoftMaxLHA
from activation.normal_activation import Sigmoid, ReLU
from gl import common_function
from loss.loss import MSELoss, CrossEntropyLoss

from bp import bp_nn
from sample.image_softmax_sample import ImageSoftMaxSample


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
    nn = bp_nn.BPNeuralNetwork(activation, last_hop_activation, loss)

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

    # sample.create_sample(train_image_root_path)
    sample.create_sample_ex(100)

    train_sx_list = sample.get_sx_list()
    train_sy_list = sample.get_sy_list()

    # 3. 训练

    # 每一层网络的神经元个数
    neuron_count_list = [10, 10]

    # 最大循环训练次数
    loop_max = 1

    # 学习效率
    rate = 0.1

    # 训练
    nn.train(train_sx_list, train_sy_list, loop_max, neuron_count_list, rate)

    # 4. 测试

    # 4.1 创建测试样本
    test_image_root_path = "./../picture/number_softmax_test"
    test_image_root_path = os.path.abspath(test_image_root_path)

    # sample.create_sample(test_image_root_path)
    sample.create_sample_ex(2)

    test_sx_list = sample.get_sx_list()
    test_sy_list = sample.get_sy_list()

    py_list = nn.predict(test_sx_list, test_sy_list)

    accuracy = common_function.calculate_accuracy(py_list, test_sy_list)

    print("\n")
    print("accuracy = %f%%" % (100 * accuracy))
    print("\n")

    print("\n")
    print("py:\n")
    count = len(py_list)

    for i in range(0, count):
        print(i)
        print("\n")
        print(py_list[i])
        print("\n")




