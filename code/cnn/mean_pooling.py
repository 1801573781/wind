"""
Function：平均汇聚
Author：lzb
Date：2021.01.11
"""

from cnn.convolution import Convolution, CVLDim

"""
class：MeanPooling 平均汇聚
说明：
1、最大汇聚，也可以看作是一种特殊的卷积
2、卷积核大小：K * K, 步长：S * S
3、卷积的计算时 mean 函数
"""


class MeanPooling(Convolution):
    """
    功能：计算 x 某一点（i, j）的卷积
    参数：
    x：输入信息
    y：待赋值的卷积结果
    i：x 的 width index
    j：x 的 height index
    d: x 的 depth index
    返回值： x 某一点（i, j）的卷积

    """

    def _cal_cvl_on_index(self, x, y, i, j, d):
        # sum 的初值 = 0
        tmp = 0

        for u in range(0, self.w_width):
            for v in range(0, self.w_height):
                # 求和
                tmp += self._x_value(x, i, j, u, v, d)

        # 3维卷积
        if CVLDim.THREE.value == self.cvl_dim:
            y[i, j, d] = tmp / (self.w_width * self.w_height)
        # 2维卷积
        else:
            y[i, j] = tmp / (self.w_width * self.w_height)
