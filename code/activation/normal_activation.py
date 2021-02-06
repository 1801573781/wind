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
    # 激活函数
    def active(self, x):
        if x <= -10:
            x = -10
        elif x >= 10:
            x = 10
        else:
            x = x

        y = 1.0 / (1.0 + math.exp(-x))

        return y

    # 求导
    def derivative(self, x):
        y = x * (1 - x)
        return y

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
    # 激活函数
    def active(self, x):
        y = max(x, 0)
        return min(y, 10)
        # return max(x, 0)

    # 求导
    def derivative(self, x):
        shape = x.shape
        if x > 0:
            y = np.ones(shape)
            return y
        else:
            y = np.zeros(shape)
            return y

    # 校正函数
    def revise(self, x):
        if x > 0:
            return dichotomy.Dichotomy.C1.value
        else:
            return dichotomy.Dichotomy.C2.value
