"""
Function：画图（将训练样本，预测样本，预测结果等画出来）
Author：lzb
Date：2021.01.01
"""

from matplotlib import pyplot as plt
import matplotlib
from enum import Enum

from activation import label


class ShowFlag(Enum):
    SHOW = 1
    NO_SHOW = 0


"""
功能：初始化画图
参数：
title：图形标题
返回值：NULL
"""


def init_draw(title):
    zh_font = matplotlib.font_manager.FontProperties(fname="./../font/SourceHanSansSC-Bold.otf")

    plt.title(title, fontproperties=zh_font)

    #  fontproperties 设置中文显示，fontsize 设置字体大小
    plt.xlabel("x轴", fontproperties=zh_font)
    plt.ylabel("y轴", fontproperties=zh_font)


"""
功能：画点
参数：
point_list：需要画的 point 的列表
c_list：point 的分类列表
返回值：NULL
"""


def draw_points(point_list, c_list):
    count = len(point_list)

    for i in range(0, count):
        # point 是一个 [n, 1] 的矩阵，我们只取前两行
        point = point_list[i]
        x = point[0][0]
        y = point[1][0]

        # c 是一个 [n, 1] 的矩阵，我们只取第一行
        point_classify = c_list[i]
        classify = point_classify[0][0]

        """
        严格来说，这段代码有问题，我们暂时先这样
        暂时先假设一共只有2类
        """
        if classify == label.Color.RED.value:
            color = "r"
        else:
            color = "g"

        plt.scatter(x, y, marker=".", c=color)


"""
功能：画线
参数：NULL
返回值：NULL
说明：需要注入一个函数，暂时不会写，空在这里
"""


def draw_line():
    pass


"""
功能：显示图像
参数：NULL
返回值：NULL
"""


def show():
    plt.show()


"""
功能：将预测结果画出来
参数：NULL      
返回值：NULL
"""


def draw_predict(title, sx_list, py_list, sample):
    # 1. 初始化
    init_draw(title)

    # 2. 画出预测的点
    draw_points(sx_list, py_list)

    # 3. 画出分割线
    sample.draw_segment()

    plt.show()
