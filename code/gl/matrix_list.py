"""
Function：matrix-list 互相转换
Author：lzb
Date：2021.02.20
"""

import numpy as np


def list_2_matrix(lst):
    """
    将 list 转换为 matrix, row = lst.len, col = 1
    :param lst: 待转换的 list
    :return: 转换后的 matrix
    """
    count = len(lst)

    arr = np.zeros([count, 1])

    for i in range(0, count):
        arr[i][0] = lst[i]

    return arr


def matrix_2_list(arr):
    """
    将 matrix 转换为 list， arr 是 [row, 1] 矩阵
    :param arr:
    :return: 转换后的 list
    """

    shape = arr.shape
    row = shape[0]

    lst = list()

    for i in range(0, row):
        lst.append(arr[i][0])

    return lst
