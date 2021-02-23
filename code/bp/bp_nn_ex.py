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
