"""
Function：Activation Function
Author：lzb
Date：2020.12.22
"""

import math
from activation import dichotomy


"""
class：Activation
说明：激活函数
"""


class Activation:
    # 激活函数，虚函数，待子类继承
    def active(self, x):
        pass

    # 校正函数
    def revise(self, x):
        pass

    # 求导
    def derivative(self):
        pass


"""
class：Sigmoid
说明：SIGMOID 激活函数
"""


class Sigmoid(Activation):
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


class ReLU(Activation):
    # 激活函数
    def active(self, x):
        return max(x, 0)

    # 求导
    def derivative(self, x):
        if x > 0:
            return 1
        else:
            return 0

    # 校正函数
    def revise(self, x):
        if x > 0:
            return dichotomy.Dichotomy.C1.value
        else:
            return dichotomy.Dichotomy.C2.value
