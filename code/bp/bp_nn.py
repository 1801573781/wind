"""
Function：BP Neural Network
Author：lzb
Date：2020.12.25
"""

import numpy as np

from nn.neural_network import NeuralNetwork


"""
class：BPNeuralNetwork，BP 神经网络
说明：
1、继承自 NeuralNetwork
2、重载 _modify_wb 函数

特别说明：
1、代码中，有将 [n, 1]（n 行，1 列） 的矩阵转为 vector（实际上是 list），这是为了看着不迷惑
2、numpy 似乎很神奇，它将 list 好像是当作了 [n, 1]（n 行，1 列） 的矩阵
"""


class BPNeuralNetwork(NeuralNetwork):
    """
    功能：修正 W，B
    参数：
    nn_y_list：神经网路计算的每一层结果，nn_y 是一个向量
    sx：训练样本的输入，sx 是一个向量
    sy：训练样本的输出，sy 是一个向量 
    返回值：NULL
    """

    def _modify_wb(self, nn_y_list, sx, sy):
        # 1. 后向传播，计算 ksi_list
        ksi_list = self.__bp(nn_y_list, sy)

        # 2. 通过 ksi_list，修正 W，B
        self.__modify_wb_by_ksi_list(ksi_list, sx, nn_y_list)

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

    def __bp(self, nn_y_list, sy):
        # 1. 初始化 ksi_list
        ksi_list = [0] * self.layer_count

        # 2. 计算最后一层 ksi

        # 2.1 计算误差(err)：最后一层的计算结果与样本输出结果的比较（计算结果 - 训练样本的输出）
        nn_y_last = nn_y_list[self.layer_count - 1]
        err = np.subtract(nn_y_last, sy)

        # 2.2 计算最后一层 ksi

        # 最后一层 ksi：ksi_last，ksi_last 是个向量
        row_last = len(nn_y_last)
        ksi_last = list()

        for i in range(0, row_last):
            # 计算ksi_last 的每个元素
            ksi_item = err[i] * self.activation.derivative(nn_y_last[i])
            ksi_last.append(ksi_item[0, 0])  # ksi_item 是一个[1, 1]的矩阵，取其元素，而不是取其矩阵本身

        ksi_list[self.layer_count - 1] = ksi_last

        # 3. 反向传播，计算：倒数第2层 ~ 第1层的 ksi
        for layer in range(self.layer_count - 2, -1, -1):
            # 当前层的 ksi
            ksi_cur = list()

            # 下一层的 ksi
            ksi_next = ksi_list[layer + 1]

            # 当前层神经网络的计算结果
            nn_y_cur = nn_y_list[layer]

            # 当前层神经元的个数
            neuron_count_cur = self.neuron_count_list[layer]

            # 下一层神经元的个数
            neuron_count_next = self.neuron_count_list[layer + 1]

            # 下一层的 w
            w = self.W[layer + 1]

            # 计算当前层的每一个 ksi
            for i in range(0, neuron_count_cur):
                # s 的计算公式为：s = sum(w[j][i] * ksi_next[j])
                s = 0
                for j in range(0, neuron_count_next):
                    s = s + w[j, i] * ksi_next[j]

                # ksi_item 的计算公式为：ksi_item = sum(w[j][i] * ksi_next[j]) * f'(y)
                # ksi_item = s * nn_y_cur[i] * (1 - nn_y_cur[i])
                ksi_item = s * self.activation.derivative(nn_y_cur[i])

                # 将 ksi_item 加入向量
                ksi_cur.append(ksi_item[0, 0])

            # 将本层计算出的 ksi 加入到 ksiList
            ksi_list[layer] = ksi_cur

        # return 计算结果
        return ksi_list

    """
    功能：修正 W，B
    参数： 
    ksi_list：每一层的 ksi 的列表，ksi 是一个向量
    sx：输入样本，sx 是一个向量
    nn_y_list：神经网络的每一层的计算结果列表，nn_y 是一个向量    
    返回值：NULL  
    """

    def __modify_wb_by_ksi_list(self, ksi_list, sx, nn_y_list):
        # 逐层修正
        for layer in range(0, self.layer_count):
            w = self.W[layer]
            b = self.B[layer]
            ksi = ksi_list[layer]

            cur_neuron_count = self.neuron_count_list[layer]

            if 0 == layer:
                pre_neuron_count = self.sx_dim
                v = sx
            else:
                pre_neuron_count = self.neuron_count_list[layer - 1]
                v = nn_y_list[layer - 1]

            for i in range(0, cur_neuron_count):
                # 计算 w[i, j]
                for j in range(0, pre_neuron_count):
                    w[i, j] = w[i, j] - self.rate * ksi[i] * v[j, 0]

                # 计算 b[i]
                b[i] = b[i] - self.rate * ksi[i]
