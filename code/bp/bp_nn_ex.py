"""
Function：分组训练的 BP 神经网络
Author：lzb
Date：2021.02.23
"""

import numpy as np

from fnn.fnn_ex import FnnEx
from gl.matrix_list import matrix_2_list, list_2_matrix


class BPFnnEx(FnnEx):
    """
    分组训练的 BP 神经网络
    """

    ''''''

    def _calc_train_para_delta(self, nn_y_list, sx, sy):
        """
        计算神经网络训练参数的 delta
        :param nn_y_list: 神经网络每一层的输出
        :param sx: 训练样本（输入）
        :param sy: 训练样本（输出）
        :return: NULL
        """

        # 1. 通过 bp 算法，计算 ksi
        ksi_list = self._bp(nn_y_list, sy)

        # 2. 通过 ksi，计算 delta w, delta b
        self._calc_delta_wb(ksi_list, sx, nn_y_list)

        # 子类需要 ksi_list
        return ksi_list

    ''''''

    def _modify_train_para(self):
        """
        根据训练参数的 delta，修正训练参数
        :return: NULL
        """

        # 修正每一层的 w，b 参数
        for layer in range(0, self._layer_count):
            self._w_layer[layer] -= self._rate * self._delta_w_layer[layer]
            self._b_layer[layer] -= self._rate * self._delta_b_layer[layer]

    ''''''

    def _bp(self, nn_y_list, sy):
        """
        后向传播，计算 ksi_list \n
        1、ksi(代表希腊字母，音：科赛)，是一个向量，每层都有，代表目标函数 E 对每一层中间输出的偏导 \n
        2、ksi_list 记录每一层的 ksi \n

        :param nn_y_list: 神经网路计算的每一层结果
        :param sy: 训练样本的输出
        :return: ksi_list
        """

        # 1. 初始化 ksi_list
        ksi_list = [0] * self._layer_count

        # 2. 计算最后一层 ksi
        ksi_last = self._calc_last_ksi(nn_y_list, sy)
        ksi_list[self._layer_count - 1] = ksi_last

        # 3. 反向传播，计算：倒数第2层 ~ 第1层的 ksi
        self._bp_ksi(nn_y_list, ksi_list)

        # return 计算结果
        return ksi_list

    ''''''

    def _calc_last_ksi(self, nn_y_list, sy):
        """
        计算最后一层 ksi
        :param nn_y_list: 神经网路计算的每一层结果
        :param sy: 训练样本的输出
        :return: 最后一层 ksi
        """

        # 1. 计算损失函数的偏导
        last_hop_y = nn_y_list[self._layer_count]
        loss_dy = self._loss.derivative_array(last_hop_y, sy)

        # 2. 计算最后一层 ksi
        nn_y_last = nn_y_list[self._layer_count - 1]
        row_last = len(nn_y_last)
        ksi_last = list()

        for i in range(0, row_last):
            # 计算ksi_last 的每个元素
            ksi_item = loss_dy[i][0] * self._last_hop_activation.derivative(last_hop_y, i) \
                       * self._activation.derivative(nn_y_last[i][0])

            ksi_last.append(ksi_item)

        ksi_last = list_2_matrix(ksi_last)

        return ksi_last

    ''''''

    def _bp_ksi(self, nn_y_list, ksi_list):
        """
        反向传播，计算：倒数第2层 ~ 第1层的 ksi
        :param nn_y_list: 神经网路计算的每一层结果
        :param ksi_list: 存储每一层的 ksi
        :return: NULL
        """

        # 反向传播
        for layer in range(self._layer_count - 2, -1, -1):
            # 1. 求解当前层激活函数的导数

            # 1.1 当前层神经网络的计算结果
            nn_y_cur = nn_y_list[layer]

            # 1.2 求导
            dy_cur_activiation = self._activation.derivative_array(nn_y_cur)

            # 1.3 将求导结果转化为对角矩阵
            diag_dy = np.diag(matrix_2_list(dy_cur_activiation))

            # 2. 下一层的 w 的转置
            w_next_T = (self._w_layer[layer + 1]).T

            # 3. 下一层的 ksi
            ksi_next = ksi_list[layer + 1]

            # 4. 计算当前层的 ksi: ksi_cur = diag_y * w_next_T, ksi_next
            ksi_cur = np.matmul(w_next_T, ksi_next)
            ksi_cur = np.matmul(diag_dy, ksi_cur)

            # 5. 将本层计算出的 ksi 加入到 ksi_list
            ksi_list[layer] = ksi_cur

    ''''''

    def _calc_delta_wb(self, ksi_list, sx, nn_y_list):
        """
        计算每一层的 delta_w, delta_b
        :param ksi_list: 每一层的 ksi 列表
        :param sx: 每一层的 ksi 列表
        :param ksi_list: 每一层的 ksi 列表
        :return:
        """

        # 因为已经通过 bp，计算出每一层的 ksi，所以，计算 delta_w, delta_b 时，就不必使用 bp 算法了，正向计算即可
        for layer in range(0, self._layer_count):
            # 该层的 ksi
            ksi = ksi_list[layer]

            if 0 == layer:
                v = sx
            else:
                v = nn_y_list[layer - 1]

            # 非常关键的计算公式
            self._delta_w_layer[layer] += np.matmul(ksi, v.T)
            self._delta_b_layer[layer] += ksi
