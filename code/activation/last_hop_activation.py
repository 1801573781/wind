"""
Function：最后一跳激活函数
Author：lzb
Date：2021.01.30
"""

import math

from activation.dichotomy import dichotomy_revise
from gl.handle_array import handle_arr, sum_arr


class LastHopActivation:
    """
    功能：最后一跳激活函数， 一个 base class\n
    说明：神经网络的最后一跳，也需要激活函数，比如 SoftMax
    """

    # 功能：训练时，最后一跳激活
    def train_activation(self, nn_y):
        """
        功能：训练时，最后一跳激活\n
        参数：\n
        nn_y：神经网络训练的输出，是一个n维数组，或者是一个数值\n
        返回值：最后一跳激活的结果（是一个n维数组，或者是一个数值）\n
        """
        return

    # 功能：预测时，最后一跳激活
    def predict_activation(self, nn_y):
        """
        功能：预测时，最后一跳激活\n
        参数：\n
        nn_y：神经网络训练的输出，是一个n维数组，或者是一个数值\n
        返回值：最后一跳激活的结果（是一个n维数组，或者是一个数值）\n
        """
        return

    # 功能：最后一跳激活函数的导数
    def derivative(self, nn_y):
        """
        功能：最后一跳激活函数的导数\n
        参数：\n
        nn_y：神经网络训练的输出，是一个n维数组，或者是一个数值\n
        返回值：最后一跳激活函数的导数（是一个n维数组，或者是一个数值）\n
        """
        return  # 暂时空实现，啥也不干


''''''


class DichotomyLHA(LastHopActivation):
    """
    功能：二分类最后一跳激活函数\n
    """

    def train_activation(self, nn_y):
        """
        功能：训练时，最后一跳激活\n
        参数：\n
        nn_y：神经网络训练的输出，是一个n维数组，或者是一个数值\n
        返回值：最后一跳激活的结果（是一个n维数组，或者是一个数值）\n
        """

        # 训练时，最后一跳不做任何处理
        return

    def predict_activation(self, nn_y):
        """
        功能：预测时，最后一跳激活\n
        参数：\n
        nn_y：神经网络训练的输出，是一个n维数组，或者是一个数值\n
        返回值：最后一跳激活的结果（是一个n维数组，或者是一个数值）\n
        """

        # 预测试，最后一跳需要修正，修正为二分类中的某一类
        handle_arr(nn_y, dichotomy_revise)


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
        SoftMaxLHA._soft_max(nn_y)

    ''''''

    def predict_activation(self, nn_y):
        """
        功能：预测时，最后一跳激活\n
        参数：\n
        nn_y：神经网络训练的输出，是一个n维数组，或者是一个数值\n
        返回值：最后一跳激活的结果（是一个n维数组，或者是一个数值）\n
        """

        # 预测时，最后一跳做 soft max 处理
        SoftMaxLHA._soft_max(nn_y)

    @staticmethod
    def _soft_max(arr):
        """
        功能：将 arr 的每个元素，求解 soft max\n
        参数：\n
        arr：多维数组\n
        返回值：NULL\n
        """
        # 1. 先将 arr 的每个元素 a，变为 a = exp(a)
        handle_arr(arr, SoftMaxLHA._exp)

        # 2. 求 arr 各元素之和
        s = [0]
        sum_arr(arr, s)

        # 3. 求解概率：将 arr 各元素 a，变为 a = a / s[0]
        handle_arr(arr, SoftMaxLHA._probability, s[0])

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
