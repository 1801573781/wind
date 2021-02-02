"""
Function：全连接神经网络训练样本，base class
Author：lzb
Date：2021.02.02
"""


class FullConnectedSample:
    """
    全连接神经网络训练样本，base class
    """

    # 样本个数
    sample_count = 0

    # 样本，输入向量维度
    sx_dim = 0

    # 样本，输出向量维度
    sy_dim = 0

    # 样本列表，输入，sx 是向量
    sx_list = 0

    # 样本列表，输出，sy 是向量
    sy_list = 0

    def get_sx_list(self):
        """
        获取输出样本输出列表
        :return: 训练样本输出列表
        """
        return self.sx_list

    def get_sy_list(self):
        """
        获取输出样本输出列表
        :return: 训练样本输出列表
        """
        return self.sy_list
