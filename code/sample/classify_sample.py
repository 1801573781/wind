"""
Function：分类-训练样本，base class
Author：lzb
Date：2021.01.01
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import random

from gl import draw

"""
class：ClassifySample，base class
"""


class ClassifySample:
    # 样本个数
    sample_count = 0

    # 样本，输入向量维度
    sx_dim = 0

    # 样本，输出向量维度
    sy_dim = 0

    # 样本，输入向量，每个元素的最大值
    sx_max = 0

    # 样本列表，输入，sx 是向量
    sx_list = 0

    # 样本列表，输出，sy 是向量
    sy_list = 0

    """
    功能：获取样本，输入
    参数：NULL    
    返回值：self.sx_list
    """

    def get_sx_list(self):
        return self.sx_list

    """
    功能：获取样本，输出
    参数：NULL    
    返回值：self.sy_list
    """

    def get_sy_list(self):
        return self.sy_list

    """
    功能：创建样本
    参数：
    sample_count：样本数量    
    sx_max：样本，输入向量，每个元素的最大值
    sx_dim：样本，输入向量的维度
    sy_dim：样本，输出向量的维度
    返回值：NULL
    """

    def create_sample(self, sample_count, sx_max, sx_dim, sy_dim):
        # 1. 初始化
        self.sample_count = sample_count
        self.sx_max = sx_max
        self.sx_dim = sx_dim
        self.sy_dim = sy_dim

        # 2. 创建训练样本，输入
        self._create_sx_list()

        # 3. 创建训练样本，输出
        self._create_sy_list()

    """
    功能：创建样本，输入
    参数：NULL
    返回值：NULL    
    """

    def _create_sx_list(self):
        # 初始化 sx_list
        self.sx_list = list()

        # 创建 sample_count 个训练样本输入，sx
        for i in range(0, self.sample_count):
            # sx 是一个 [sx_dim, 1] 的矩阵
            sx = np.empty([self.sx_dim, 1])
            for j in range(0, self.sx_dim):
                """
                1、默认采用随机数创建
                2、random.random() 是介于 (0, 1) 之间的一个随机数（记为 r）
                3、r 减去 0.5，是为了构建随机的正负数，其范围是 (-0.5, 0,5)
                4、所以，需要乘以2，再乘以 max
                """
                sx[j][0] = (random.random() - 0.5) * 2 * self.sx_max[j]

            self.sx_list.append(sx)

    """
    功能：创建样本，输出
    参数：NULL
    返回值：NULL
    说明：这是一个虚函数，待子类重载  
    """

    def _create_sy_list(self):
        pass

    """
    功能：画出样本
    参数：NULL
    返回值：NULL    
    """

    def draw_sample(self, title):
        # 初始化
        draw.init_draw(title)

        # 画样本点
        draw.draw_points(self.sx_list, self.sy_list)

        # 画分割（线）
        self.draw_segment()

        # 显示图像
        draw.show()

    """
    功能：画分割（线）
    参数：NULL
    返回值：NULL
    说明：这是一个虚函数，待子类重载
    """

    def draw_segment(self):
        pass

    """
    创建固定样本
    """

    def create_sample_stub(self):
        # 2. 创建训练样本，输入
        self._create_sx_list_stub()

        # 3. 创建训练样本，输出
        self._create_sy_list()

    """
    功能：创建固定样本，输入
    参数：NULL
    返回值：NULL    
    """

    def _create_sx_list_stub(self):
        pass
