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


def dichotomy_revise(*args):
    """
    功能：二分类时，值的校正\n
    参数：\n
    *args：待校正的值\n
    返回值：校正后的值\n
    """

    # 待校验的值，位于 args[0][0]
    x = args[0][0]

    if x >= (Dichotomy.C1.value + Dichotomy.C2.value) / 2:
        return Dichotomy.C1.value
    else:
        return Dichotomy.C2.value
