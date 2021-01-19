"""
Function：通用函数
Author：lzb
Date：2021.01.01
"""

import numpy as np
import random


"""
功能：计算正确率
参数：
py_list：预测结果列表
sy_list：样本结果列表
返回值：NULL
"""


def calculate_accuracy(py_list, sy_list):
    # 1. 合法性校验
    c_py = len(py_list)
    c_sy = len(sy_list)

    if (c_py != c_sy) or (0 == c_py):
        print("\n错误的参数，c_py = %d, c_sy = %d\n" % (c_py, c_sy))
        return -1

    # 计算正确率
    count = c_py
    accuracy = 0

    for i in range(0, count):
        py = py_list[i]
        sy = sy_list[i]

        if py[0, 0] == sy[0, 0]:
            accuracy = accuracy + 1
        else:
            pass

    return accuracy / count


"""
功能：随机化3维数组
参数：
width：3维数组的 width
height：3维数组的 height
depth：3维数组的 depth
返回值：随机数，数组
说明：因为 random.random() 的范围是 0~1 之间，所以 减去 0.5，使得范围变成 -0.5~0.5 之间

"""


def rand_array_3(width, height, depth):
    array = np.zeros([width, height, depth])

    for i in range(0, width):
        for j in range(0, height):
            for k in range(0, depth):
                array[i, j, k] = random.random() - 0.5

    return array
