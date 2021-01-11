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
    功能：计算卷积
    参数：
    x：输入信息
    y：待赋值的卷积结果
    返回值：NULL
    说明：重载 _cal_cvl 函数，采用 max 算法
    """

    def _cal_cvl(self, x, y):
        # 卷积的 width，height
        y_width = y.shape[0]
        y_height = y.shape[1]

        # 分配一个临时数组
        tmp_list = list()

        # 计算卷积 y
        for i in range(0, y_width):
            for j in range(0, y_height):
                # 3维卷积
                if CVLDim.THREE.value == self.cvl_dim:
                    for d in range(0, self.w_depth):
                        for u in range(0, self.w_width):
                            for v in range(0, self.w_height):
                                # 计算每一个值
                                tmp = self.w[u, v, d] * x[(i * self.s + u), (j * self.s + v), d]
                                tmp_list.append(tmp)

                        # 等于最大值
                        y[i, j, d] = max(tmp_list)

                # 2维卷积
                else:
                    for u in range(0, self.w_width):
                        for v in range(0, self.w_height):
                            # 计算每一个值
                            tmp = self.w[u, v] * x[(i * self.s + u), (j * self.s + v)]
                            tmp_list.append(tmp)

                    # 等于最大值
                    y[i, j] = max(tmp_list)