"""
Function：定义两个分类的值
Author：lzb
Date：2021.01.01
"""

from enum import Enum


class Dichotomy(Enum):
    # 类别1
    C1 = 1

    # 类别2
    C2 = 0

    # 修正
    def revise(self, x):
        if x >= (self.C1 + self.C2) / 2:
            return self.C1
        else:
            return self.C2
