"""
Function：循环神经网络
Author：lzb
Date：2021.02.27

2021.02.11，开始写循环神经网络，一直到今天，还没有搞定
"""

import os
import pickle

import numpy as np

from bp.bp_nn_ex import BPFnnEx
from gl.array_string import array_2_string
from gl.common_function import unserialize_train_para
from gl.matrix_list import matrix_2_list


class RnnEx(BPFnnEx):
    """
    循环神经网络 \n
    特别说明： \n
    1、h(t) = f(u * h(t - 1) + (w * x + b)) \n
    2、隐藏层也有多层，那么对于 u * h(t - 1) 而言，它该是几层呢？ \n
    3、先假设 u * h(t - 1) 只作用于第1层？ \n
    """

    # 每一层 u 参数
    _u_layer = None

    # 每一层 b 参数的 delta
    _delta_u_layer = None

    # 隐藏层 h(t) 输出，时间序列
    _hidden_out_sequence = None

    # 训练样本(输入)，时间序列
    _sx_list = None

    # RNN 只作用于第1层的标记
    _rnn_layer_0 = True

    ''''''

    def _init_train_para(self):
        """
        初始化训练参数
        :return: NULL
        """

        # 通过反序列化，初始化 w，b, u
        if self._init_from_unserialization:
            file_path = os.path.dirname(__file__) + "/../gl/train_para/"
            self._w_layer, self._b_layer, self._u_layer = \
                unserialize_train_para(file_path, self._layer_count, u_flag=True)
        # 通过随机数，初始化 w, b, u
        else:
            # 先调用父类的 _init_train_para
            super()._init_train_para()

            # 初始化 _u_layer（每一层 u 参数）
            self._u_layer = list()

            # 虽然本 class 暂时只实现第一层的 rnn，但是 u 参数还是每一层都做个初始化
            for i in range(0, self._layer_count):
                u = self._alpha_para * np.random.random((self._neuron_count_list[i], self._neuron_count_list[i]))
                self._u_layer.append(u)

    ''''''

    def _pre_train(self):
        """
        每一组训练之前预准备工作
        :return: NULL
        """

        # 重新初始化 隐藏层 h(t) 输出，时间序列
        self._hidden_out_sequence = list()

        # 重新初始化 训练样本(输入)，时间序列
        self._sx_list = list()

    ''''''

    def _pre_predict(self):
        """
        预测前的准备工作
        :return: NULL
        """

        # 重新初始化 隐藏层 h(t) 输出，时间序列
        self._hidden_out_sequence = list()

        # 重新初始化 训练样本(输入)，时间序列
        self._sx_list = list()

    ''''''

    def _calc_nn(self, sx):
        """
        计算整个网络的输出
        :param sx: 神经网络的输入
        :return: 整个神经网络，每一层的输出
        """

        # 调用父类计算神经网络的输出
        nn_y_list = super()._calc_nn(sx)

        # 将神经网络的输出记录下来（时间序列）
        self._hidden_out_sequence.append(nn_y_list)

        # 将神经网络的训练样本(输入)记录下来（时间序列）
        self._sx_list.append(sx)

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

        # 时间序列长度
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

    def _init_train_para_delta(self):
        """
        初始化训练参数的 delta
        :return: NULL
        """

        # 1. 调用父类，初始化 delta_w, delta_b
        super()._init_train_para_delta()

        # 2. 初始化 delta_u
        self._delta_u_layer = list()

        for i in range(0, self._layer_count):
            # _delta_u, _u 维度相同，初始值为 0
            _delta_u = np.zeros(self._u_layer[i].shape)
            self._delta_u_layer.append(_delta_u)

    ''''''

    def _calc_train_para_delta(self, nn_y_list, sx, sy):
        """
        计算神经网络训练参数的 delta
        :param nn_y_list: 神经网络每一层的输出
        :param sx: 训练样本（输入）
        :param sy: 训练样本（输出）
        :return: NULL
        """

        # 1. 调用父类，计算纵向 delta w, delta b
        ksi_list = super()._calc_train_para_delta(nn_y_list, sx, sy)

        # 2. BPTT 算法，计算 eta_list (暂时只计算第一层（layer = 0）的 eta)
        eta_list = self._bptt(ksi_list, layer=0)

        # 3. 根据 eta_list 计算 delta u，并且再度计算 delta w, delta b
        self._calc_delta_wbu(eta_list, layer=0)

    ''''''

    def _modify_train_para(self):
        """
        根据训练参数的 delta，修正训练参数
        :return: NULL
        """

        # 调用父类函数，修正每一层的 w, b
        super()._modify_train_para()

        # 修正第0层的 u 参数（暂时只修正第0层）
        for layer in range(0, 0):
            self._u_layer[layer] -= self._rate * self._delta_u_layer[layer]

    ''''''

    def _bptt(self, ksi_list, layer=0):
        """
        随时间反向传播（backpropagation through time, bttt），计算沿着时间轴的 eta_list
        :param ksi_list: 当前时间轴的每一层的 ksi 列表
        :param layer: 计算某一层的 bptt, layer 默认值是0
        :return: eta_list
        """

        # 1. 当前时刻
        cur_t = len(self._hidden_out_sequence)

        # 如果当前是 t0 时刻（cur_t = 1），则无须 bptt
        if cur_t <= 1:
            return None

        # 2. eta_list 初始化
        eta_list = [0] * (cur_t - 1)

        # 3. 按照时间反向传播，计算 eta

        # 3.1 eta_last，等于该层纵向的 ksi
        eta_last = ksi_list[layer]

        # 3.2 该层（layer）的 u 参数的转置（u.T）
        uT = self._u_layer[layer].T

        # 3.3 反向计算该层（layer）的 eta
        eta_pre = eta_last
        for t in range((cur_t - 2), -1, -1):
            # 本时刻隐藏层的输出
            hidden_out = self._hidden_out_sequence[t][layer]
            # 本时刻隐藏层输出的导数
            dh = self._activation.derivative_array(hidden_out)
            # 将导数变为对角线矩阵
            diag_dh = np.diag(matrix_2_list(dh))
            # 计算 eta
            eta = np.matmul(uT, eta_pre)
            eta = np.matmul(diag_dh, eta)

            # 存储 delta
            eta_list[t] = eta

            # 递归（循环）
            eta_pre = eta

        # 返回 eta_list
        return eta_list

    ''''''

    def _calc_delta_wbu(self, eta_list, layer=0):
        """
        根据 eta_list 计算 delta u，并且再度计算 delta w, delta b
        :param eta_list: 随时间反向传播（backpropagation through time, bttt），计算出沿着时间轴的 eta_list
        :return: NULL
        """

        # 如果 eta_list = None, 说明是 t0 时刻，此时还不需要计算 detla u/w/b
        if eta_list is None:
            return

        # 当前时刻
        cur_t = len(self._hidden_out_sequence)

        t = cur_t - 2

        dw = np.matmul(eta_list[t], self._sx_list[t].T)
        db = eta_list[t]
        du = np.matmul(eta_list[t], self._hidden_out_sequence[t][layer].T)

        # 计算 delta w, delta b, delta u
        self._delta_w_layer[layer] += dw
        self._delta_b_layer[layer] += db
        self._delta_u_layer[layer] += du

    ''''''

    def _create_train_para_string(self):
        """
        将训练参数转化为 string
        :return: 练参数转化后的 string
        """

        # 这段代码写的不好，与父类有大量重复，后续再优化吧

        # 训练参数字符串
        train_para_str = ""

        # 再补上 u 的字符串
        for layer in range(0, self._layer_count):
            # loop
            train_para_str += "loop ＝ %d\n\n" % layer

            # w
            train_para_str += "w%d:\n\n" % layer
            train_para_str += array_2_string(self._w_layer[layer])

            # b
            train_para_str += "\n\n"
            train_para_str += "b%d:\n\n" % layer
            train_para_str += array_2_string(self._b_layer[layer])

            # u
            train_para_str += "\n\n"
            train_para_str += "u%d:\n\n" % layer
            train_para_str += array_2_string(self._u_layer[layer])

            # 换行
            train_para_str += "\n\n"

        return train_para_str

    ''''''

    def _serialize_train_para(self):
        """
        将训练参数序列化到文件
        :return: NULL
        """

        # 先调用父类，序列化 w, b
        super()._serialize_train_para()

        # 再序列化 u
        file_path = os.path.dirname(__file__) + "/../" + self._para_file_path

        for layer in range(0, self._layer_count):
            # u 参数文件名
            u_file_name = file_path + "u%d" % layer
            # 序列化 u
            pickle.dump(self._u_layer[layer], open(u_file_name, 'wb'))

    ''''''

    def stub_set_para(self, neuron_count_list, w_layer, b_layer, u_layer):
        """
        设置神经网络参数：w, b, u
        :param neuron_count_list: 神经网络层数
        :param w_layer: 每一层 w 参数 列表
        :param b_layer: 每一层 b 参数 列表
        :param u_layer: 每一层 u 参数 列表
        :return: NULL
        """

        # 每一层神经元的数量(Neuron Count)
        self._neuron_count_list = neuron_count_list

        # 神经网络层数
        self._layer_count = len(w_layer)

        # 每一层 w 参数
        self._w_layer = w_layer

        # 每一层 b 参数
        self._b_layer = b_layer

        # 每一层 u 参数
        self._u_layer = u_layer
