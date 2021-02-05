"""
Function：Loss Function（损失函数）
Author：lzb
Date：2021.01.29
"""

import numpy as np
import math

from gl.handle_array import sum_arr, get_arr_item, handle_arr_ex


class Loss:
    """
    损失函数(base class)
    """

    def loss(self, nn_y, sy):
        """
        功能：计算损失值\n
        参数：\n
        nn_y：神经网络的输出，n维数组\n
        sy：训练样本的输出，n维数组\n
        返回值：损失值\n
        """

        lss = [0]

        self._loss(nn_y, sy, lss)

        return lss[0]

    ''''''

    def derivative_index(self, nn_y, sy, index):
        """
        功能：损失函数求导\n
        参数：\n
        nn_y：神经网络的输出，n维数组\n
        sy：训练样本的输出，n维数组\n
        index：n维数组的索引,1维数组，比如 [2, 3, 4]，表示 arr[2][3][4]\n
        返回值：损失函数的导数\n
        """

        pass

    ''''''

    def derivative(self, nn_y, sy):
        """
        功能：损失函数求导\n
        参数：\n
        nn_y：神经网络的输出，n维数组\n
        sy：训练样本的输出，n维数组\n
        返回值：损失函数的导数\n
        """

        pass

    ''''''

    def _loss(self, nn_y, sy, lss):
        """
        功能：计算损失值\n
        参数：\n
        nn_y：神经网络的输出，n维数组\n
        sy：训练样本的输出，n维数组\n
        lss：损失值，是一个 1行1列的数组，因为单独一个数，python 不支持引用传递\n
        返回值：NULL\n
        说明：递归计算损失值\n
        """

        # 省略参数的合法性判断

        # 数组的维度
        shape = nn_y.shape
        dim = len(shape)

        # 第1维元素的个数
        count = shape[0]

        # 如果是1维，则开始计算
        if 1 == dim:
            for i in range(0, count):
                lss[0] += self._element_loss(nn_y[i], sy[i])

            return
        # 如果不是1维，则需要递归：
        else:
            for i in range(0, count):
                a = nn_y[i]
                b = sy[i]

                self._loss(a, b, lss)

    ''''''

    def _element_loss(self, nn_y_item, sy_item):
        """
        功能：计算数组的某一个具体元素的损失值\n
        参数：\n
        a：神经网络的输出，最后已经递归到1维数组\n
        b：训练样本的输出，最后已经递归到1维数组\n
        返回值：损失值\n
        """

        pass


''''''


class MSELoss(Loss):
    """
    均方差损失函数
    """

    def _element_loss(self, nn_y_item, sy_item):
        """
        功能：计算数组的某一个具体元素的损失值\n
        参数：\n
        a：神经网络的输出，最后已经递归到1维数组\n
        b：训练样本的输出，最后已经递归到1维数组\n
        返回值：损失值\n
        """

        e_loss = 0.5 * (nn_y_item - sy_item) ** 2
        return e_loss

    ''''''

    def derivative_index(self, nn_y, sy, index):
        """
        功能：损失函数求导\n
        参数：\n
        nn_y：神经网络的输出，n维数组\n
        sy：训练样本的输出，n维数组\n
        index：n维数组的索引,1维数组，比如 [2, 3, 4]，表示 arr[2][3][4]\n
        返回值：损失函数的导数\n
        """

        nn_y_item = get_arr_item(nn_y, index)
        sy_item = get_arr_item(sy, index)

        dy = nn_y_item - sy_item
        return dy

    ''''''

    def derivative(self, nn_y, sy):
        """
        功能：损失函数求导\n
        参数：\n
        nn_y：神经网络的输出，n维数组\n
        sy：训练样本的输出，n维数组\n
        返回值：损失函数的导数\n
        """

        dy = np.subtract(nn_y, sy)
        return dy


class CrossEntropyLoss(Loss):
    """
    交叉熵损失函数
    """

    def _element_loss(self, nn_y_item, sy_item):
        """
        功能：计算数组的某一个具体元素的损失值\n
        参数：\n
        a：神经网络的输出，最后已经递归到1维数组\n
        b：训练样本的输出，最后已经递归到1维数组\n
        返回值：损失值\n
        """

        e_loss = (-1) * math.log(nn_y_item) * sy_item
        return e_loss

    ''''''

    def derivative(self, nn_y, sy):
        """
        损失函数求导
        :param nn_y: 神经网络的输出，n维数组
        :param sy: 训练样本的输出，n维数组
        :return: 损失函数的导数
        """

        """
        # 初始化
        dy = np.asarray(nn_y)
        arr_list = [nn_y, sy]

        handle_arr_ex(arr_list, dy, CrossEntropyLoss._derivative)
        """

        dy = np.subtract(nn_y, 1)

        return dy

    ''''''

    def derivative_index(self, nn_y, sy, index):
        """
        功能：损失函数求导\n
        参数：\n
        nn_y：神经网络的输出，n维数组\n
        sy：训练样本的输出，n维数组\n
        index：n维数组的索引,1维数组，比如 [2, 3, 4]，表示 arr[2][3][4]\n
        返回值：损失函数的导数\n
        """

        nn_y_item = get_arr_item(nn_y, index)
        # dy = math.log(nn_y_item) - 1
        dy = nn_y_item - 1

        return dy

    ''''''

    @staticmethod
    def _derivative(dy, index, *args):
        """
        CrossEntropyLoss 的求导
        :param dy: CrossEntropyLoss 的导数。dy 已经被迭代为 1维数组
        :param index: dy 的 index
        :param args: args[0], nn_y(迭代最后的数值)；args[1], sy（迭代到最后的数值）
        :return: NULL
        """

        nn_y_item = args[0][0]
        # sy_item = args[0][1]  # 这个值，不需要

        # dy[index] = math.log(nn_y_item) - 1
        dy[index] = nn_y_item - 1


''''''


def test():
    """
    功能：测试损失函数\n
    参数：NULL\n
    返回值：NULL\n
    """

    loss = MSELoss()

    nn_y = np.asarray([[[50.113265], [93.826943], [115.373675]],
                       [[59.189640], [119.609671], [104.736651]],
                       [[42.784839], [90.116834], [67.911661]]])

    sy = np.asarray([[[0.496053], [0.142468], [0.692607]],
                     [[-0.152569], [0.746855], [0.665359]],
                     [[-0.078141], [0.187796], [0.004870]]])

    print("\nMSELoss\n")

    err = loss.loss(nn_y, sy)

    print("\nloss = %f\n" % err)

    dy = loss.derivative_index(nn_y, sy, [0, 0, 0])

    print("\ndy = %f\n" % dy)

    dy = loss.derivative(nn_y, sy)

    print("\ndy = \n")
    print(dy)

    print("\nCrossEntropyLoss\n")

    loss = CrossEntropyLoss()

    err = loss.loss(nn_y, sy)

    print("\nloss = %f\n" % err)

    dy = loss.derivative_index(nn_y, sy, [0, 0, 0])

    print("\ndy = %f\n" % dy)

    dy = loss.derivative(nn_y, sy)

    print("\ndy = \n")
    print(dy)

