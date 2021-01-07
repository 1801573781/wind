"""
Function：感知器
Author：lzb
Date：2020.12.22
"""

import numpy as np

from gl import errorcode

from nn.neural_network import NeuralNetwork

"""
class：Perceptron 感知器
说明：
1、继承自 NeuralNetwork
2、重载 _valid、_modify_wb 函数
"""


class Perceptron(NeuralNetwork):
    """
    功能：校验训练样本是否正确
    参数：NULL
    返回值：错误码
    """

    def _valid(self):
        err = super()._valid()

        if errorcode.SUCCESS != err:
            return err

        # 感知器只能是1层
        layer_count = len(self.neuron_count_list)

        if 1 != layer_count:
            return errorcode.FAILED

        # 感知器的输出只能是1维向量
        sy_dim = self.neuron_count_list[layer_count - 1]

        if 1 != sy_dim:
            return errorcode.FAILED

        return errorcode.SUCCESS

    """
    功能：修正 W，B
    参数：
    nn_y_list：神经网路计算的每一层结果，nn_y 是一个向量
    sx：训练样本的输入，sx 是一个向量
    sy：训练样本的输出，sy 是一个向量 
    返回值：NULL
    特别说明：
    1、感知器非常简单，对于 sy，b 而言，实际上就是1个值而已
    2、所以代码中出现了 err[0, 0]、b[0, 0] 这样的直接写死的下标    
    """

    def _modify_wb(self, nn_y_list, sx, sy):
        # 逐层修正
        for layer in range(0, self.layer_count):
            w = self.W[layer]
            b = self.B[layer]

            nn_y_last = nn_y_list[self.layer_count - 1]
            err = np.subtract(nn_y_last, sy)

            # 修正 w, b
            if 0 != err[0, 0]:
                # 修正 w
                for j in range(0, self.sx_dim):
                    w[0, j] = w[0, j] - self.rate * err[0, 0] * sx[j, 0]

                # 修正 b
                b[0, 0] = b[0, 0] - self.rate * err[0, 0]
            else:
                pass