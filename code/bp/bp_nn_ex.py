"""
Function：分组训练的 BP 神经网络
Author：lzb
Date：2021.02.23
"""
from fnn.fnn_ex import FNNEx


class BPFNNEx(FNNEx):
    """
    分组训练的 BP 神经网络
    """

    ''''''

    def _calc_train_para_delta(self, nn_y_list, sx, sy, delta_list):
        """
        计算神经网络训练参数的 delta
        :param nn_y_list: 神经网络每一层的输出
        :param sx: 训练样本（输入）
        :param sy: 训练样本（输出）
        :param delta_list: 训练参数 delta 列表
        :return: NULL
        """
        pass

    ''''''

    def _modify_train_para(self, delta_list):
        """
        根据 delta_list，修正训练参数
        :param delta_list: 训练参数的 delta 列表
        :return: NULL
        """
        pass

    ''''''

    """
    功能：后向传播，计算 ksi_list
    参数：
    nn_y_list：神经网路计算的每一层结果，nn_y 是一个向量    
    sy：训练样本的输出，sy 是一个向量 
    返回值：ksi_list
    说明：
    1、ksi(代表希腊字母，音：科赛)，是一个向量，每层都有，代表目标函数 E 对每一层中间输出的偏导
    2、ksi_list 记录每一层的 ksi
    """

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

        # 2.1 计算损失函数的偏导
        last_hop_y = nn_y_list[self._layer_count]
        loss_dy = self._loss.derivative_array(last_hop_y, sy)

        # 2.2 计算最后一层 ksi

        # 最后一层 ksi：ksi_last，ksi_last 是个向量
        nn_y_last = nn_y_list[self._layer_count - 1]
        row_last = len(nn_y_last)
        ksi_last = list()

        for i in range(0, row_last):
            # 计算ksi_last 的每个元素
            ksi_item = loss_dy[i][0] * self._last_hop_activation.derivative(last_hop_y, i) \
                       * self._activation.derivative(nn_y_last[i][0])

            ksi_last.append(ksi_item)

        ksi_list[self._layer_count - 1] = ksi_last

        # 3. 反向传播，计算：倒数第2层 ~ 第1层的 ksi
        for layer in range(self._layer_count - 2, -1, -1):
            # 当前层的 ksi
            ksi_cur = list()

            # 下一层的 ksi
            ksi_next = ksi_list[layer + 1]

            # 当前层神经网络的计算结果
            nn_y_cur = nn_y_list[layer]

            # 当前层神经元的个数
            neuron_count_cur = self._neuron_count_list[layer]

            # 下一层神经元的个数
            neuron_count_next = self._neuron_count_list[layer + 1]

            # 下一层的 w
            w = self._w_layer[layer + 1]

            # 计算当前层的每一个 ksi
            for i in range(0, neuron_count_cur):
                # s 的计算公式为：s = sum(w[j][i] * ksi_next[j])
                s = 0
                for j in range(0, neuron_count_next):
                    s = s + w[j, i] * ksi_next[j]

                # ksi_item 的计算公式为：ksi_item = sum(w[j][i] * ksi_next[j]) * f'(y)
                ksi_item = s * self._activation.derivative(nn_y_cur[i])

                # 将 ksi_item 加入向量
                # ksi_cur.append(ksi_item[0])
                ksi_cur.append(ksi_item)

            # 将本层计算出的 ksi 加入到 ksiList
            ksi_list[layer] = ksi_cur

        # return 计算结果
        return ksi_list
