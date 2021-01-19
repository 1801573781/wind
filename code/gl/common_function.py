"""
Function：通用函数
Author：lzb
Date：2021.01.01
"""

import numpy as np
import random

from gl.common_enum import ArrayDim

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
        print("\n错误的参数，c_py = %d, c_sy = %d\n" & (c_py, c_sy))
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


"""
功能：数组转换为字符串
参数：
arr：数组    
返回值：数组转换后的字符串
"""


def array_string(arr):
    # 数组的维度
    dim = len(arr.shape)

    # 1维数组
    if ArrayDim.ONE.value == dim:
        return array_1_string(arr)
    # 2维数组
    elif ArrayDim.TWO.value == dim:
        return array_2_string(arr)
    # 3维数组（剩下的都当作3维数组，超过3维的，暂时也当作3维数组）
    else:
        return array_3_string(arr)


"""
功能：1维数组转换为字符串
参数：
arr：1维数组    
返回值：1维数组转换后的字符串
"""


def array_1_string(arr):
    # 数组 width
    width = arr.shape[0]

    # 构建字符串

    # 字符串起始为 "["
    arr_str = "["

    # 构建每一个元素
    for i in range(0, width):
        arr_str += "%f" % arr[i]

        # 每一个元素都做这种判断，不必在乎性能（arr 数据量没有达到海量，也没有海量重复，这点性能损失，不算什么）
        if i < (width - 1):
            arr_str += ",\n"
        # 最后1个，没有 ",\n"
        else:
            pass

    # 字符串终止为 "]"
    arr_str += "]"

    return arr_str


"""
功能：2维数组转换为字符串
参数：
arr：2维数组    
返回值：2维数组转换后的字符串
"""


def array_2_string(arr):
    # 数组 width，height
    width = arr.shape[0]
    height = arr.shape[1]

    # 构建字符串

    # 字符串起始为 "["
    arr_str = "["

    # 一直构建到倒数第2个
    for i in range(0, width):
        # 每一行的起始为"["
        arr_str += "["

        for j in range(0, height):
            arr_str += "%f" % arr[i, j]

            # 每一个元素后面，需要加上 ", "，除了最后一个元素
            if j < (height - 1):
                arr_str += ", "
            # 最后一个元素，没有 ", "
            else:
                pass

        # 每一行的终止为 "]"
        arr_str += "]"

        # 每一行后面需要加上 ", \n"，除了最后一行
        if i < (width - 1):
            arr_str += ", \n"
        # 最后一行终止没有 ", \n"
        else:
            pass

    # 字符串终止为 "]"
    arr_str += "]"

    return arr_str


"""
功能：3维数组转换为字符串
参数：
arr：3维数组    
返回值：3维数组转换后的字符串
"""


def array_3_string(arr):
    # 数组 width，height, depth
    width = arr.shape[0]
    height = arr.shape[1]
    depth = arr.shape[2]

    # 构建字符串

    # 字符串起始为 "["
    arr_str = "["

    # 一直构建到倒数第2个
    for i in range(0, width):
        # 每一行的起始为"["
        arr_str += "["

        for j in range(0, height):
            # 每一列的起始为 "["
            arr_str += "["

            for k in range(0, depth):
                arr_str += "%f" % arr[i, j, k]

                # 每一个元素后面，需要加上 ", "，除了最后一个元素
                if k < (depth - 1):
                    arr_str += ", "
                # 最后一个元素，没有 ", "
                else:
                    pass

            # 每一 depth 的终止为 "]"
            arr_str += "]"

            # 每一列后面需要加上 ", \n"，除了最后一行
            if j < (height - 1):
                arr_str += "], "
            # 最后一行终止没有 ", \n"
            else:
                pass

        # 每一行后面需要加上 ", \n"，除了最后一行
        if i < (width - 1):
            arr_str += ", \n"
        # 最后一行终止没有 ", \n"
        else:
            pass

    # 字符串终止为 "]"
    arr_str += "]"

    return arr_str




