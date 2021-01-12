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
    功能：计算卷积
    参数：
    x：输入信息
    y：待赋值的卷积结果
    返回值：NULL
    说明：重载 _cal_cvl 函数，采用 mean 算法
    """

    def _cal_cvl(self, x, y):
        # 1. 卷积的 width，height
        y_width = y.shape[0]
        y_height = y.shape[1]

        # 2. 平均卷积 y
        for i in range(0, y_width):
            for j in range(0, y_height):
                # 3维卷积
                if CVLDim.THREE.value == self.cvl_dim:
                    for d in range(0, self.w_depth):
                        # sum 的初值 = 0
                        tmp = 0

                        for u in range(0, self.w_width):
                            for v in range(0, self.w_height):
                                # 求和
                                tmp += x[(i * self.s + u), (j * self.s + v), d]

                        # 等于平均值
                        y[i, j, d] = tmp / (self.w_width * self.w_height)
                # 2维卷积
                else:
                    # sum 的初值 = 0
                    tmp = 0

                    for u in range(0, self.w_width):
                        for v in range(0, self.w_height):
                            # 求和
                            tmp += x[(i * self.s + u), (j * self.s + v)]

                            # 等于平均值
                    y[i, j] = tmp / (self.w_width * self.w_height)


