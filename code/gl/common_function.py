"""
Function：通用函数
Author：lzb
Date：2021.01.01
"""

import pickle
import time

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
    # 合法性校验
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


''''''


def get_local_time():
    """
    获取当时的本地时间
    :return: 当时的本地时间
    """

    localtime = time.localtime()
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", localtime)

    return localtime


''''''


def unserialize_train_para(file_path, layer_count, u_flag=False):
    """
    从文件中反序列化 w，b, u 参数
    :param file_path: 序列化文件所在路径
    :param layer_count: 神经网络层数
    :param u_flag: 是否反序列化 u 参数
    :return: w_layer，b_layer，u_layer
    """

    # 初始化
    w_layer = list()
    b_layer = list()
    u_layer = list()

    # 逐层反序列化
    for i in range(0, layer_count):
        # w
        file_name = file_path + "w%d" % i
        w = pickle.load(open(file_name, 'rb'))
        w_layer.append(w)

        # b
        file_name = file_path + "b%d" % i
        b = pickle.load(open(file_name, 'rb'))
        b_layer.append(b)

        # u
        if u_flag:
            file_name = file_path + "u%d" % i
            u = pickle.load(open(file_name, 'rb'))
            u_layer.append(u)

    # return 反序列化结果
    if u_flag:
        return w_layer, b_layer, u_layer
    else:
        return w_layer, b_layer
