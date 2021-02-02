"""
Function：分类-训练样本，两根直线 分割的样本
Author：lzb
Date：2021.01.01
"""

import numpy as np
from matplotlib import pyplot as plt

from activation import dichotomy
from sample.points_sample import PointsSample

"""
class：StraightLineSample
说明：
1、重载两个函数： _create_sy_list, draw_segment
2、对于 StraightLineSample 而言，其输入只可能是2个维度的向量，其输出只可能是1个维度的向量
3、所以代码中，针对输入输出的向量维度，有很多写死的地方
"""


class TwoLineSample(PointsSample):
    # a0, b0
    a0 = 1
    b0 = 0.5

    # h1
    a1 = 1
    b1 = -0.5

    """
    功能：重载父类的 _create_sy_list
    参数：NULL
    返回值：NULL
    说明：
    1、对于 SinSample 而言，其输入只可能是2个维度的向量，其输出只可能是1个维度的向量
    2、所以代码中，针对输入输出的向量维度，直接写死
    """

    def _create_sy_list(self):
        # 1. 初始化
        self.sy_list = list()

        # 2. 构建 sy_list，sy 是分类 C1 or C2
        for i in range(0, self.sample_count):
            # sx 是一个 [2, 1] 的矩阵
            sx = self.sx_list[i]
            x0 = sx[0][0]  # 对应到坐标系的 x
            x1 = sx[1][0]  # 对应到坐标系的 y

            # sy 是一个 [1, 1] 的矩阵
            sy = np.empty([1, 1])

            sl0_x0 = self.a0 * x0 + self.b0
            sl1_x1 = self.a1 * x0 + self.b1

            # 比较
            if (x1 >= sl0_x0) or (x1 <= sl1_x1):
                sy[0][0] = dichotomy.Dichotomy.C1.value
            else:
                sy[0][0] = dichotomy.Dichotomy.C2.value

            self.sy_list.append(sy)

    """
    功能：重载父类的 draw_segment
    参数：NULL
    返回值：NULL    
    """

    def draw_segment(self):
        # 1. 绘制两根直线
        """
        # plt.axhline(self.h0, color='blue', linewidth=1.0)
        # plt.axhline(self.h1, color='blue', linewidth=1.0)
        """

        x = np.arange(-(self.sx_max[0]), (self.sx_max[0] + 0.5))
        y = self.a0 * x + self.b0
        plt.plot(x, y, color='blue', linewidth=1.0)

        x = np.arange(-(self.sx_max[0]), (self.sx_max[0] + 0.5))
        y = self.a1 * x + self.b1
        plt.plot(x, y, color='blue', linewidth=1.0)

    """
    功能：创建固定样本，输入
    参数：NULL
    返回值：NULL    
    """

    def _create_sx_list_stub(self):
        #
        self.sx_max = list()
        self.sx_max.append(1)
        self.sx_max.append(1)

        # 样本数量
        self.sample_count = 5

        # 初始化 sx_list
        self.sx_list = list()

        sx = np.empty([2, 1])
        sx[0, 0] = 0
        sx[1, 0] = 0.8
        self.sx_list.append(sx)

        sx = np.empty([2, 1])
        sx[0, 0] = 0
        sx[1, 0] = -0.8
        self.sx_list.append(sx)

        sx = np.empty([2, 1])
        sx[0, 0] = 0.5
        sx[1, 0] = 0.5
        self.sx_list.append(sx)

        sx = np.empty([2, 1])
        sx[0, 0] = -0.5
        sx[1, 0] = -0.5
        self.sx_list.append(sx)

        sx = np.empty([2, 1])
        sx[0, 0] = 0
        sx[1, 0] = 0
        self.sx_list.append(sx)
