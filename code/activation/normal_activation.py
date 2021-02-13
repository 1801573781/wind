"""
Function：Activation Function
Author：lzb
Date：2020.12.22
"""

import numpy as np
import math
from activation import dichotomy

"""
class：NormalActivation
说明：激活函数
"""


class NormalActivation:
    # 最大值
    _max_value = -1

    # 激活函数，虚函数，待子类继承
    def active(self, x):
        pass

    # 校正函数
    def revise(self, x):
        pass

    # 求导
    def derivative(self, x):
        pass


"""
class：Sigmoid
说明：SIGMOID 激活函数
"""


class Sigmoid(NormalActivation):
    # 构造函数
    def __init__(self, max_value=20):
        self._max_value = max_value

    # 激活函数
    def active(self, x):
        if x <= -self._max_value:
            x = -self._max_value
        elif x >= self._max_value:
            x = self._max_value
        else:
            x = x

        y = 1.0 / (1.0 + math.exp(-x))

        return y

    # 求导
    def derivative(self, x):
        dy = x * (1 - x)
        return dy

    # 校正函数
    def revise(self, x):
        if x > 0.5:
            return dichotomy.Dichotomy.C1.value
        else:
            return dichotomy.Dichotomy.C2.value


"""
class：ReLU
说明：ReLU 激活函数
"""


class ReLU(NormalActivation):
    # 构造函数
    def __init__(self, max_value=-1):
        self._max_value = max_value

    # 激活函数
    def active(self, x):
        y = max(x, 0)

        if self._max_value > 0:
            y = min(y, self._max_value)

        return y

    # 求导
    def derivative(self, x):
        shape = x.shape
        if x > 0:
            dy = np.ones(shape)
            return dy
        else:
            dy = np.zeros(shape)
            return dy

    # 校正函数
    def revise(self, x):
        if x > 0:
            return dichotomy.Dichotomy.C1.value
        else:
            return dichotomy.Dichotomy.C2.value


# noinspection SpellCheckingInspection
"""
class：Tanh
说明：Tanh 激活函数
"""


# noinspection SpellCheckingInspection
class Tanh(NormalActivation):

    # 构造函数
    def __init__(self, max_value=20):
        """
        构造函数
        :param max_value:由于需要计算 exp(x)，所以 x 不能太大
        """
        self._max_value = max_value

    # 激活函数
    def active(self, x):
        """
        激活函数，tanh
        :param x: 待激活的变量
        :return: tanh(x)
        """
        if x <= -self._max_value:
            x = -self._max_value
        elif x >= self._max_value:
            x = self._max_value
        else:
            x = x

        y = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

        return y

    # 激活函数求导
    def derivative(self, x):
        """
        激活函数 tanh(x) 求导数
        :param x: tanh(x)待求导的变量
        :return: tanh(x)的导数
        """
        y = self.active(x)

        dy = 1 - y ^ 2

        return dy

    # 激活函数求导
    @staticmethod
    def derivative_ex(y):
        """
        激活函数 tanh(x) 求导数
        :param y: y = tanh(x)，激活函数的结果
        :return: tanh(x)的导数
        """
        dy = 1 - y ^ 2

        return dy

    # 校正函数
    def revise(self, x):
        if x > 0:
            return dichotomy.Dichotomy.C1.value
        else:
            return dichotomy.Dichotomy.C2.value
