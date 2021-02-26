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
    _sample_count = 0

    # 样本，输入向量维度
    _sx_dim = 0

    # 样本，输出向量维度
    _sy_dim = 0

    # 样本列表，输入，sx 是向量
    _sx_list = 0

    # 样本列表，输出，sy 是向量
    _sy_list = 0

    ''''''

    def get_sx_list(self):
        """
        获取输出样本输出列表
        :return: 训练样本输出列表
        """
        return self._sx_list

    ''''''

    def get_sy_list(self):
        """
        获取输出样本输出列表
        :return: 训练样本输出列表
        """
        return self._sy_list

    ''''''

    def get_sx_group(self):
        """
        将训练样本（输入），分组。默认是1个训练样本，1组
        :return: 训练样本（输入）分组
        """

        return FullConnectedSample._get_sample_group(self._sx_list)

    ''''''

    def get_sy_group(self):
        """
        将训练样本（输出），分组。默认是1个训练样本，1组
        :return: 训练样本（输出）分组
        """

        return FullConnectedSample._get_sample_group(self._sy_list)

    ''''''

    @staticmethod
    def _get_sample_group(sample_list):
        """
        将训练样本，分组。默认是1个训练样本，1组
        :param sample_list:
        :return: 训练样本分组
        """

        sample_group = list()

        count = len(sample_list)

        for i in range(0, count):
            sample = sample_list[i]
            # 1个样本组成1个list
            sl = [sample]
            # 1样本组成的list，是1个分组
            sample_group.append(sl)

        return sample_group
