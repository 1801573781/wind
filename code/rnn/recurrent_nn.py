"""
Function：循环神经网络
Author：lzb
Date：2021.02.10

特别说明：明天（2021.02.11）是除夕！
"""

import numpy as np

from nn.feedforward_neural_network import FNN


class RecurrentNN(FNN):
    """
    循环神经网络，继承自 NeuralNetwork
    特别说明： \n
    1、h(t) = f(u * h(t - 1) + (w * x + b)) \n
    2、隐藏层也有多层，那么对于 u * h(t - 1) 而言，它该是几层呢？ \n
    3、先假设 u * h(t - 1) 只作用于第1层？ \n
    """

    # 每一层 u 参数，u 是个 matrix（BP 网络） or 3维数组（卷积网络）
    _U = None

    # 隐藏层 h(t) list
    _HT_list = None

    # 当前的 t
    _cur_t = 0

    # RNN 只作用于第1层的标记
    _rnn_layer_0 = True

    # 初始化 self._U
    def _init_other_para(self):
        """
        初始化 self._U
        :return: NULL
        """
        super()._init_other_para()

        # 初始化 self._U
        self._U = list()

        for i in range(0, self._layer_count):
            u = np.random.random((self._neuron_count_list[i], self._neuron_count_list[i]))
            self._U.append(u)

    # 每一轮训练之前预准备工作
    def _pre_train(self):
        """
        每一轮训练之前预准备工作
        :return: NULL
        """

        # _HT_list 重新初始化
        _HT_list = list()

        # 当前的 t，重新初始化
        _cur_t = 0

    # 计算整个网络的输出
    def _calc_nn(self, sx):
        """
        计算整个网络的输出
        :param sx: 神经网络的输入
        :return: 整个神经网络，每一层的输出
        """

        nn_y_list = super()._calc_nn(sx)

        self._HT_list.append(nn_y_list)

        return nn_y_list

    ''''''

    def _calc_recurrent(self, layer):
        """
        计算循环神经网络， u * h(t - 1) ，默认值是 0
        :param layer: 层数
        :return: u * h(t - 1)
        """

        # 如果只计算第1层，如果层数超过第1层，则直接 return 0
        if self._rnn_layer_0:
            if layer > 0:
                return 0

        # 其他情形，则计算 recurrent

        count = len(self._HT_list)

        # 0 == count，意味着是 t0 时刻，此时还不存在前一个状态
        if 0 == count:
            return 0
        # 此时意味着，存在前一状态
        else:
            nn_y_list_pre_time = self._HT_list[count - 1]  # 前一状态的神经网络各层的输出
            nn_y_pre_time = nn_y_list_pre_time[layer]  # 前一状态的神经网络第 layer 层的输出
            u = self._U[layer]  # 该层的 u 参数
            uy = np.matmul(u, nn_y_pre_time)

            return uy

