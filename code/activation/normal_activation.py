"""
Function：Activation Function
Author：lzb
Date：2020.12.22
"""

import numpy as np
import math

from gl.handle_array import handle_arr_ex

"""
class：NormalActivation
说明：激活函数
"""


class NormalActivation:
    # 最大值
    _max_value = -1

    def active(self, x):
        """
        激活函数，虚函数，待子类重载
        :param x: 待激活的数
        :return: 激活后的值
        """
        pass

    ''''''

    def active_array(self, arr):
        """
        针对一个数组，激活
        :param arr: 待激活的数组
        :return: 激活后的结果
        """

        active_arr = np.zeros(arr.shape)

        arr_list = [arr]
        handle_arr_ex(arr_list, active_arr, self._derivative_array_callback)

        return active_arr

    ''''''

    def _active_array_callback(self, *args):
        """
        针对一个数组激活，回调函数
        :param args: args[0][0] 为 x
        :return: self.active(x)
        """

        x = args[0][0]
        return self.active(x)

    ''''''

    def derivative(self, x):
        """
        针对一个数求导，虚函数，待子类重载
        :param x: 待求导的数
        :return: 求导后的值
        """
        pass

    ''''''

    def derivative_array(self, arr):
        """
        针对一个数组求导
        :param arr: 待求导的数组（数组的维度是可变的/未知的）
        :return: 数组求导的结果
        """

        dy_arr = np.zeros(arr.shape)

        arr_list = [arr]
        handle_arr_ex(arr_list, dy_arr, self._derivative_array_callback)

        return dy_arr

    ''''''

    def _derivative_array_callback(self, *args):
        """
        针对一个数组求导，回调函数
        :param args: args[0][0] 为 x
        :return: self.derivative(x)
        """
        x = args[0][0]
        return self.derivative(x)


class Sigmoid(NormalActivation):
    """
    SIGMOID 激活函数
    """

    def __init__(self, max_value=20):
        """
        构建函数
        :param max_value: 由于涉及到 exp(x) 的计算，所以 x 不能太大，也不宜太小
        """
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


class ReLU(NormalActivation):
    """
    ReLU 激活函数
    """

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
        if x > 0:
            return 1
        else:
            return 0


# noinspection SpellCheckingInspection
class Tanh(NormalActivation):
    """
    Tanh 激活函数
    """

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
