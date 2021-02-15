"""
Function：循环神经网络
Author：lzb
Date：2021.02.10

特别说明：明天（2021.02.11）是除夕！
"""

import numpy as np

from fnn.feedforward_nn import FNN


class RecurrentNN(FNN):
    """
    循环神经网络，继承自 NeuralNetwork
    特别说明： \n
    1、h(t) = f(u * h(t - 1) + (w * x + b)) \n
    2、隐藏层也有多层，那么对于 u * h(t - 1) 而言，它该是几层呢？ \n
    3、先假设 u * h(t - 1) 只作用于第1层？ \n
    """

    # 每一层 u 参数，u 是个 matrix（BP 网络） or 3维数组（卷积网络）
    _u_layer = None

    # 隐藏层 h(t) 输出，时间序列
    _hidden_out_sequence = None

    # 当前的 t
    _cur_t = 0

    # RNN 只作用于第1层的标记
    _rnn_layer_0 = True

    ''''''

    def _init_other_para(self):
        """
        初始化 self._U
        :return: NULL
        """
        super()._init_other_para()

        # 初始化 self._U
        self._u_layer = list()

        for i in range(0, self._layer_count):
            u = np.random.random((self._neuron_count_list[i], self._neuron_count_list[i]))
            self._u_layer.append(u)

    ''''''

    def _pre_train(self):
        """
        每一轮训练之前预准备工作
        :return: NULL
        """

        # _HT_list 重新初始化
        _HT_list = list()

        # 当前的 t，重新初始化
        _cur_t = 0

    ''''''

    def _calc_nn(self, sx):
        """
        计算整个网络的输出
        :param sx: 神经网络的输入
        :return: 整个神经网络，每一层的输出
        """

        nn_y_list = super()._calc_nn(sx)

        self._hidden_out_sequence.append(nn_y_list)

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

        T = len(self._hidden_out_sequence)

        # 0 == T，意味着是 t0 时刻，此时还不存在前一个状态
        if 0 == T:
            return 0
        # 此时意味着，存在前一状态
        else:
            nn_y_list_pre_time = self._hidden_out_sequence[T - 1]  # 前一状态的神经网络各层的输出
            nn_y_pre_time = nn_y_list_pre_time[layer]  # 前一状态的神经网络第 layer 层的输出
            u = self._u_layer[layer]  # 该层的 u 参数
            uy = np.matmul(u, nn_y_pre_time)

            return uy

    ''''''

    def _modify_fnn_para(self, nn_y_list, sx, sy):
        """
        修正神经网络的参数，w, b, u
        :param nn_y_list: 神经网路计算的每一层结果，nn_y 是一个向量
        :param sx: 训练样本的输入，sx 是一个向量
        :param sy: 训练样本的输出，sy 是一个向量
        :return: NULL
        """
        # 1. 后向传播，计算当前时间轴的每一层的 ksi
        ksi_list = self.__bp(nn_y_list, sy)

        # 2. 随时间反向传播（backpropagation through time, bttt），计算沿着时间轴的 delta_list

    ''''''

    def _bptt(self, ksi_list, layer=0):
        """
        随时间反向传播（backpropagation through time, bttt），计算沿着时间轴的 delta_list
        :param ksi_list: 当前时间轴的每一层的 ksi 列表
        :param layer: 计算某一层的 bptt
        :return: delta_list
        """

        # delta_list 初始化
        delta_list = list()

        # 当前时刻
        T = len(self._hidden_out_sequence)
