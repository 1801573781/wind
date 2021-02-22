"""
Function：分组训练的 FNN
Author：lzb
Date：2021.02.22

特别说明：
1、以前写的代码，class FNN，没有考虑分组训练
2、现在考虑分组训练，暂时先重写代码
3、以前的代码（FNN） 暂时先保留，待 class FNNEx 完善以后，再删除原先的代码
"""

import numpy as np

import time

from gl import errorcode
from gl.array_string import array_2_string

from activation.normal_activation import Sigmoid
from activation.last_hop_activation import DichotomyLHA
from loss.loss import MSELoss


class FNNEx:
    """
    分组训练的 FNN
    """

    # 神经网络输入样本，向量维度
    _sx_dim = 0

    # 神经网络输出样本，向量维度
    _sy_dim = 0

    # 神经网络层数
    _layer_count = 0

    # 每一层神经元的数量
    _neuron_count_list = None

    # 每一层 w 参数，w 是个 matrix（BP 网络） or 3维数组（卷积网络）
    _w_layer = None

    # 每一层 b 参数，b 是个 vector（BP 网络） or 2维数组（卷积网络）
    _b_layer = None

    # 每一层 w 参数的 shape list（除了卷积网络，这个参数没有意义）
    _w_shape_layer = None

    # 样本数量
    _sample_count = 0

    # 训练样本分组列表(输入)
    _sx_group_list = None

    # 训练样本分组列表(输出)
    _sy_group_list = None

    # 循环训练的最大次数
    _loop_max = 1

    # 学习效率
    _rate = 0

    # 激活函数对象（class Activation 的实例）
    _activation = Sigmoid()

    # 最后一跳激活函数对象（class LastHopActivation 的实例）
    _last_hop_activation = DichotomyLHA()

    # 损失函数
    _loss = MSELoss()

    def __init__(self, activation=None, last_hop_activation=None, loss=None):
        """
        构造函数
        :param activation: 激活函数对象
        :param last_hop_activation: 后一跳激活函数对象
        :param loss: 损失函数对象
        """

        if activation is not None:
            self._activation = activation

        if last_hop_activation is not None:
            self._last_hop_activation = last_hop_activation

        if loss is not None:
            self._loss = loss

    ''''''

    def train(self, sx_group_list, sy_list, loop_max, neuron_count_list, rate, w_shape_list=None):
        """
        功能：神经网络训练\n
        参数：\n
        sx_list：训练样本输入列表\n
        sy_list：训练样本输出列表\n
        loop_max：循环训练的最大次数 \n
        neuron_count_list：每一层神经元数量(对于卷积网络，这个参数没有意义)\n
        rate：学习效率 \n
        activation：激活函数对象\n
        last_hop_activation：最后一跳激活函数对象\n
        loss：损失函数对象\n
        w_shape_list：每一层 w 参数的 shape list（除了卷积网络，这个参数没有意义）\n
        返回值：错误码\n
        """

        # 1. 成员变量赋值
        self._sx_list = sx_group_list
        self._sy_list = sy_list
        self._loop_max = loop_max
        self._rate = rate

        # 如果是卷积网络，这个参数没有意义（如果是卷积网络，直接传入 None 即可）
        self._neuron_count_list = neuron_count_list

        # 如果不是卷积网络，这个参数，没有意义（如果不是卷积网络，直接传入默认值即可）
        self._w_shape_layer = w_shape_list

        # 2. 校验
        err = self._valid()
        if errorcode.SUCCESS != err:
            print("\nvalid error, errcode = %d\n" % err)
            return err

        # 3. 初始化 w, b，及其他参数
        self._init_other_para()

        # 4. 训练
        return self._train()
