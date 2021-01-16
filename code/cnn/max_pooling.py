"""
Function：最大汇聚
Author：lzb
Date：2021.01.11
"""

from cnn.convolution import Convolution, CVLDim

"""
class：MaxPooling 最大汇聚
说明：
1、最大汇聚，也可以看作是一种特殊的卷积
2、卷积核大小：K * K, 步长：S * S
3、卷积的计算时 max 函数
"""


class MaxPooling(Convolution):
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
        # 分配一个临时数组
        tmp_list = list()

        for u in range(0, self.w_width):
            for v in range(0, self.w_height):
                tmp_list.append(self._x_value(x, i, j, u, v, d))

        # 3维卷积
        if CVLDim.THREE.value == self.cvl_dim:
            y[i, j, d] = max(tmp_list)
        # 2维卷积
        else:
            y[i, j] = max(tmp_list)

