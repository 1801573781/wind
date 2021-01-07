"""
Function：通用函数
Author：lzb
Date：2021.01.01
"""


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
