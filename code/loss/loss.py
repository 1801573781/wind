"""
Function：Loss Function（损失函数）
Author：lzb
Date：2021.01.29
"""

import numpy as np

"""
class：Loss (base)
"""


class Loss:
    """
    功能：计算损失值
    参数：
    nn_y：神经网络的输出，n维数组
    sy：训练样本的输出，n维数组
    返回值：损失值
    """

    def loss(self, nn_y, sy):
        lss = [0]

        self._loss(nn_y, sy, lss)

        lss[0] = self._correction(lss[0])

        return lss[0]

    """
    功能：损失函数求导
    参数： 
    nn_y：神经网络的输出，n维数组
    sy：训练样本的输出，n维数组
    index：n维数组的索引,1维数组，比如 [2, 3, 4]，表示 arr[2][3][4]
    返回值：损失函数的导数
    """

    def derivative(self, nn_y, sy, index):
        pass

    """
    功能：计算损失值
    参数：
    nn_y：神经网络的输出，n维数组
    sy：训练样本的输出，n维数组
    lss：损失值，是一个 1行1列的数组，因为单独一个数，python 不支持引用传递
    返回值：NULL
    说明：递归计算损失值
    """

    def _loss(self, nn_y, sy, lss):
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

    """
    功能：修正损失函数的计算结果
    参数：
    v：带修正的数值    
    返回值：修正后的值
    说明：均方差损失函数，最后需要乘以 0.5。为了效率，是最后再乘以 0.5，而不是每个计算步骤都乘以 0.5
    """

    def _correction(self, v):
        # 默认实现，是不改变
        return v

    """
    功能：计算数组的某一个具体元素的损失值
    参数：
    a：神经网络的输出，最后已经递归到1维数组
    b：训练样本的输出，最后已经递归到1维数组    
    返回值：损失值    
    """

    def _element_loss(self, a, b):
        pass

    """
    功能：求数组的某个索引的值
    参数： 
    arr：n维数组
    index：n维数组的索引,1维数组，比如 [2, 3, 4]，表示 arr[2][3][4]
    返回值：数组的某个索引的值， arr[index]
    """

    def _get_value(self, arr, index):
        # 索引所包含元素的个数
        count = len(index)

        # 用递归方法，获取 arr[index]
        a = arr[index[0]]
        for i in range(1, count):
            a = a[index[i]]

        return a


"""
class：MSELoss
说明：均方差损失函数
"""


class MSELoss(Loss):
    """
    功能：计算数组的某一个具体元素的损失值
    参数：
    a：神经网络的输出，最后已经递归到1维数组
    b：训练样本的输出，最后已经递归到1维数组
    返回值：损失值
    """

    def _element_loss(self, a, b):
        c = (a - b) ** 2
        return c

    """
    功能：修正损失函数的计算结果
    参数：
    v：带修正的数值
    返回值：修正后的值
    说明：均方差损失函数，最后需要乘以 0.5。为了效率，是最后再乘以 0.5，而不是每个计算步骤都乘以 0.5
    """

    def _correction(self, v):
        # 默认实现，是不改变
        return v * 0.5

    """
    功能：损失函数求导
    参数： 
    nn_y：神经网络的输出，n维数组
    sy：训练样本的输出，n维数组
    index：n维数组的索引,1维数组，比如 [2, 3, 4]，表示 arr[2][3][4]
    返回值：损失函数的导数
    """

    def derivative(self, nn_y, sy, index):
        a = self._get_value(nn_y, index)
        b = self._get_value(sy, index)

        c = a - b
        return c


"""
功能：测试损失函数
参数：NULL 
返回值：NULL
"""


def test():
    mse = MSELoss()

    nn_y = np.asarray([[[50.113265], [93.826943], [115.373675]],
                       [[59.189640], [119.609671], [104.736651]],
                       [[42.784839], [90.116834], [67.911661]]])

    sy = np.asarray([[[0.496053], [0.142468], [0.692607]],
                     [[-0.152569], [0.746855], [0.665359]],
                     [[-0.078141], [0.187796], [0.004870]]])

    err = mse.loss(nn_y, sy)

    print("\nloss = %f\n" % err)

    dvt = mse.derivative(nn_y, sy, [0, 0, 0])

    print("\ndvt = %f\n" % dvt)
