"""
Function：最后一跳激活函数
Author：lzb
Date：2021.01.30
"""

import math
import numpy as np

from activation.dichotomy import dichotomy_revise
from gl.handle_array import sum_arr, handle_arr_ex


class LastHopActivation:
    """
    功能：最后一跳激活函数， 一个 base class \n
    说明：神经网络的最后一跳，也需要激活函数，比如 SoftMax \n
    """

    def train_activation(self, nn_y):
        """
        训练时，最后一跳激活
        :param nn_y: 神经网络训练的输出，是一个n维数组，或者是一个数值
        :return: 最后一跳激活的结果（是一个n维数组，或者是一个数值）
        """

        # 默认实现，将神经网络的输出，复制一份输出
        last_hop_y = np.asarray(nn_y)

        return last_hop_y

    def predict_activation(self, nn_y):
        """
        预测时，最后一跳激活
        :param nn_y: 神经网络训练的输出，是一个n维数组，或者是一个数值
        :return: 最后一跳激活的结果（是一个n维数组，或者是一个数值）
        """

        # 默认实现，将神经网络的输出，复制一份输出
        last_hop_y = np.asarray(nn_y)

        return last_hop_y

    ''''''

    def predict_revise(self, lha_y, revise_strong=False):
        """
        预测时，最后一跳激活之后，再修正
        :param lha_y: 最后一跳激活之后的输出
        :param revise_strong: 强修正 flag
        :return: 最后一跳激活之后，再修正的结果
        """

        # 默认实现，将 lha_y 复制一份输出
        lhr_y = np.asarray(lha_y)

        return lhr_y

    ''''''

    def derivative(self, last_hop_y, index):
        """
        预测时，最后一跳激活函数的导数
        :param last_hop_y: 最后一跳的输出，是一个n维数组，或者是一个数值
        :param index: 最后一跳的输出的索引，是一个向量
        :return: 最后一跳激活函数的导数（是一个n维数组，或者是一个数值）
        """

        # 默认实现
        return 1


''''''


class DichotomyLHA(LastHopActivation):
    """
    功能：二分类最后一跳激活函数\n
    """

    def predict_revise(self, lha_y, revise_strong=False):
        """
        预测时，最后一跳激活之后，再修正
        :param lha_y: 最后一跳激活之后的输出
        :param revise_strong: 强修正 flag
        :return: 最后一跳激活之后，再修正的结果
        """

        # 将神经网络的输出，复制一份输出
        lhr_y = np.zeros(lha_y.shape)

        # 预测试，最后一跳需要修正，修正为二分类中的某一类
        arr_list = [lha_y]
        handle_arr_ex(arr_list, lhr_y, dichotomy_revise)

        return lhr_y

    ''''''

    '''
    def predict_activation(self, nn_y):
        """
        功能：预测时，最后一跳激活\n
        参数：\n
        nn_y：神经网络训练的输出，是一个n维数组，或者是一个数值\n
        返回值：最后一跳激活的结果（是一个n维数组，或者是一个数值）\n
        """

        # 将神经网络的输出，复制一份输出
        last_hop_y = np.zeros(nn_y.shape)

        # 预测试，最后一跳需要修正，修正为二分类中的某一类
        arr_list = [nn_y]
        handle_arr_ex(arr_list, last_hop_y, dichotomy_revise)

        return last_hop_y
    '''

    def derivative(self, last_hop_y, index):
        """
        预测时，最后一跳激活函数的导数 \n
        对于 DichotomyLHA 而言，训练时，其最后一跳并没有做任何激活处理，所以导数为1 \n
        :param last_hop_y: 最后一跳的输出，是一个n维数组，或者是一个数值
        :param index: 最后一跳的输出的索引，是一个向量
        :return: 最后一跳激活函数的导数（是一个n维数组，或者是一个数值）
        """

        return 1


''''''


class SoftMaxLHA(LastHopActivation):
    """
    功能：SoftMax 最后一跳激活函数\n
    """

    ''''''

    def train_activation(self, nn_y):
        """
        功能：训练时，最后一跳激活\n
        参数：\n
        nn_y：神经网络训练的输出，是一个n维数组，或者是一个数值\n
        返回值：最后一跳激活的结果（是一个n维数组，或者是一个数值）\n
        """

        # 训练时，最后一跳做 soft max 处理
        return SoftMaxLHA._soft_max(nn_y)

    ''''''

    def predict_activation(self, nn_y):
        """
        功能：预测时，最后一跳激活\n
        参数：\n
        nn_y：神经网络训练的输出，是一个n维数组，或者是一个数值\n
        返回值：最后一跳激活的结果（是一个n维数组，或者是一个数值）\n
        """

        # 预测时，最后一跳做 soft max 处理
        return SoftMaxLHA._soft_max(nn_y)

    ''''''

    @staticmethod
    def _soft_max(arr):
        """
        功能：将 arr 的每个元素，求解 soft max\n
        参数：\n
        arr：多维数组\n
        返回值：NULL\n
        """

        # 1. exp_arr = exp(arr)
        exp_arr = np.zeros(arr.shape)
        arr_list = [arr]
        handle_arr_ex(arr_list, exp_arr, SoftMaxLHA._exp)

        # 2. 求 arr 各元素之和
        s = [0]
        sum_arr(exp_arr, s)

        # 3. 求解概率
        last_hop_arr = np.zeros(arr.shape)
        arr_list = [exp_arr]
        handle_arr_ex(arr_list, last_hop_arr, SoftMaxLHA._probability, s[0])

        return last_hop_arr

    ''''''

    @staticmethod
    def _exp(*args):
        """
        功能：求解 e^x
        参数：\n
        args：args[0][0] 为 x
        返回值：e^x
        """

        x = args[0][0]
        return math.exp(x)

    @staticmethod
    def _probability(*args):
        """
        功能：求解概率，P = a / s
        参数：\n
        args[0][0]：a，数组 arr 中的某一个元素\n
        args[0][0]：s，数组所有元素之和
        返回值：a / s\n
        """

        a = args[0][0]
        s = args[0][1]

        return a / s

    ''''''

    def predict_revise(self, lha_y, revise_strong=False):
        """
        预测时，最后一跳激活之后，再修正
        :param lha_y: 最后一跳激活之后的输出
        :param revise_strong: 强修正 flag
        :return: 最后一跳激活之后，再修正的结果
        """

        if revise_strong:
            return SoftMaxLHA._strong_revise(lha_y)
        else:
            return SoftMaxLHA._weak_revise(lha_y)

    ''''''
    @staticmethod
    def _strong_revise(lha_y):
        """
        强修正：最大值修正为 1，其余值修正为 0
        :param lha_y:
        :return: 修正后的值
        """

        # 不搞那么复杂了，因为是 softmax，所以可以肯定 lha_y 是一个 [row, 1] 矩阵（只有1列）

        # 将神经网络的输出，复制一份输出
        lhr_y = np.zeros(lha_y.shape)

        # 获取 lha 最大值的索引
        max_index = SoftMaxLHA._get_max_index(lha_y)

        # 将 lhr_y 该索引位置赋值为1
        lhr_y[max_index][0] = 1

        return lhr_y

    ''''''

    @staticmethod
    def _get_max_index(lha_y):
        """
        获取 lha_y 最大值的索引
        :param lha_y: [row, 1] 矩阵
        :return: lha_y 最大值的索引
        """

        max_value = 0
        max_index = 0
        row = lha_y.shape[0]

        # 可以肯定，lha_y 每个值都大于0
        for r in range(0, row):
            if lha_y[r][0] > max_value:
                max_value = lha_y[r][0]
                max_index = r

        return max_index

    ''''''

    @staticmethod
    def _weak_revise(lha_y):
        """
        弱修正：只有大于一定的值，才修正为1，只有小于一定的值，才修正为0
        :param lha_y: [row, 1] 矩阵
        :return: 修正后的值
        """

        # 不搞那么复杂了，因为是 softmax，所以可以肯定 lha_y 是一个 [row, 1] 矩阵（只有1列）

        # 将神经网络的输出，复制一份输出
        lhr_y = np.zeros(lha_y.shape)

        row = lha_y.shape[0]

        max_value = 0.9
        min_value = 0.1

        for r in range(0, row):
            if lha_y[r][0] >= max_value:
                lhr_y[r][0] = 1
            elif lha_y[r][0] <= min_value:
                lhr_y[r][0] = 0
            else:
                lhr_y[r][0] = lha_y[r][0]

        return lhr_y

    ''''''

    def derivative(self, last_hop_y, index):
        """
        预测时，最后一跳激活函数的导数 \n
        对于 SoftMaxLHA 而言，训练时，其最后一跳虽然做了 SoftMax 处理，但是为了计算效率，将其导数的计算合并到交叉熵那里了，所以导数也为1 \n
        :param last_hop_y: 最后一跳的输出，是一个n维数组，或者是一个数值
        :param index: 最后一跳的输出的索引，是一个向量
        :return: 最后一跳激活函数的导数（是一个n维数组，或者是一个数值）
        """

        return 1
