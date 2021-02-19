"""
Function：循环神经网络
Author：lzb
Date：2021.02.10

特别说明：明天（2021.02.11）是除夕！
"""

import numpy as np

from bp.bp_nn import BPFNN
from gl.matrix_list import matrix_2_list, list_2_matrix


class RecurrentNN(BPFNN):
    """
    循环神经网络，继承自 BPFNN
    特别说明： \n
    1、h(t) = f(u * h(t - 1) + (w * x + b)) \n
    2、隐藏层也有多层，那么对于 u * h(t - 1) 而言，它该是几层呢？ \n
    3、先假设 u * h(t - 1) 只作用于第1层？ \n
    """

    # 每一层 u 参数，u 是个 matrix（BP 网络） or 3维数组（卷积网络）
    _u_layer = None

    # 隐藏层 h(t) 输出，时间序列
    _hidden_out_sequence = None

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

        # _hidden_out_sequence 重新初始化
        self._hidden_out_sequence = list()

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
        :param nn_y_list: 神经网路，最新时刻计算的每一层结果，nn_y 是一个向量
        :param sx: 训练样本的输入，sx 是一个向量
        :param sy: 训练样本的输出，sy 是一个向量
        :return: NULL
        """
        # 1. 后向传播，计算当前时间轴的每一层的 ksi
        ksi_list = self._bp(nn_y_list, sy)

        # 2. 随时间反向传播（backpropagation through time, bttt），计算沿着时间轴的 delta_list
        cur_t = len(self._hidden_out_sequence)
        delta_list = self._bptt(cur_t, ksi_list, 0)

        # 3. 修正 w, b, u

        # 3.1 先修正所有层 w, b
        self._modify_wb_by_ksi_list(ksi_list, sx, nn_y_list)

        # 3.2 修正第0层的 w, b, u（其中，w, b 是再修正一次）
        self._modify_uwb_by_delta_list(delta_list, sx, layer=0)

    ''''''

    def _bptt(self, cur_t, ksi_list, layer=0):
        """
        随时间反向传播（backpropagation through time, bttt），计算沿着时间轴的 delta_list
        :param cur_t: 当前时刻
        :param ksi_list: 当前时间轴的每一层的 ksi 列表
        :param layer: 计算某一层的 bptt, layer 默认值是0
        :return: delta_list
        """

        # 如果当前是 t0 时刻（cur_t = 1），则无须 bptt
        if cur_t <= 1:
            return None

        # delta_list 初始化
        delta_list = [0] * (cur_t - 1)

        # delta 初始化
        delta = list_2_matrix(ksi_list[layer])

        # 获取该层（layer）的 u.T
        uT = self._u_layer[layer].T

        # 反向计算 delta
        for t in range((cur_t - 2), -1, -1):
            # 上一时刻的输出
            hidden_out_pre = self._hidden_out_sequence[t][layer]
            # 上一时刻输出的导数
            dh = self._activation.derivative_array(hidden_out_pre)
            # 将导数变为对角线矩阵
            diag_dh = np.diag(matrix_2_list(dh))
            # 计算 delta
            delta = np.matmul(uT, delta)
            delta = np.matmul(diag_dh, delta)

            # 存储 delta
            delta_list[t] = delta

        return delta_list

    ''''''

    def _modify_uwb_by_delta_list(self, delta_list, sx, layer=0):
        """
        修正 w, b, u
        :param delta_list: 沿着时间轴的 delta_list
        :param sx: 训练输入样本
        :param layer: 神经网络层数，默认是0层
        :return: NULL
        """

        # 如果 delta_list 是 None（也就意味着是 t0 时刻），则无须修正
        if delta_list is None:
            return

        # 获取第 layer（0）层的 w, b, u
        w = self._w_layer[layer]
        b = self._b_layer[layer]
        u = self._u_layer[layer]

        # 当前时刻
        cur_t = len(self._hidden_out_sequence)

        # 偏导初始化
        dw = np.zeros(self._w_layer[layer].shape)
        db = np.zeros(self._b_layer[layer].shape)
        du = np.zeros(self._u_layer[layer].shape)

        # 偏导按照时间序列相加
        for t in range(0, cur_t - 1):
            dw = dw + np.matmul(delta_list[t], self._sx_list[t].T)
            db = db + delta_list[t]
            du = du + np.matmul(delta_list[t], self._hidden_out_sequence[t][layer].T)

        # 修正 w, b, u
        w = w - self._rate * dw
        b = b - self._rate * db
        u = u - self._rate * du



