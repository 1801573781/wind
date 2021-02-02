"""
Function：分类-训练样本，sin(x) 分割的样本
Author：lzb
Date：2021.01.01
"""

import numpy as np
from matplotlib import pyplot as plt

from activation import dichotomy
from sample.points_sample import PointsSample

"""
class：SinSample
说明：
1、重载两个函数： _create_sy_list, draw_segment
2、对于 SinSample 而言，其输入只可能是2个维度的向量，其输出只可能是1个维度的向量
3、所以代码中，针对输入输出的向量维度，有很多写死的地方
"""


class SinSample(PointsSample):
    # 欧米咖：sin(wx)
    omega = 2 * np.pi

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

            # 计算 sin(x0)
            sin_x0 = self.sx_max[1] * np.sin(self.omega * x0)

            # 比较
            if x1 >= sin_x0:
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
        # 1. 绘制 sin(x) 图像
        x = np.linspace(-self.sx_max[0], self.sx_max[0], 300)
        y = self.sx_max[1] * np.sin(self.omega * x)
        plt.plot(x, y, color='blue', linewidth=1.0)
