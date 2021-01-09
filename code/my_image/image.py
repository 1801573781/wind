"""
Function：图像处理
Author：lzb
Date：2021.01.09
"""

import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np

from gl import errorcode

"""
class：Image 图像处理
"""


class MyImage:
    """
    功能：将一个 RGB 图像转为灰度图像
    参数：NULL    
    返回值：
    gray：图像的灰度数组（是一个2维数组）
    err：错误码
    """

    def gray_image(self, file_name):
        # 1. 读取图像

        # 如果出异常了，那就是文件不存在
        try:
            # my_image 是个三维数组
            image = mpimg.imread(file_name)
        except BaseException as err_msg:
            print(err_msg)
            return 0, errorcode.FAILED

        # 2. 转化为灰度

        # 图像宽度和高度
        width = image.shape[0]
        height = image.shape[1]

        # 构建灰度图像2维数组
        gray = np.zeros([width, height])

        # 转换成灰度
        for i in range(0, width):
            for j in range(0, height):
                # rgb 是一个3元组
                rgb = image[i, j]

                # 灰度转换公式为：0.299 * R + 0.587 * G + 0.114 * B
                gray[i, j] = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

        return gray, errorcode.SUCCESS

    """
    功能：显示一个图像
    参数：NULL    
    返回值：错误码
    """

    def show_file(self, file_name):
        # 如果出异常了，那就是文件不存在
        try:
            # my_image 是个三维数组
            image = mpimg.imread(file_name)
            self.show_data(image)
            return errorcode.SUCCESS
        except BaseException as err_msg:
            print(err_msg)
            return errorcode.FAILED

    """
    功能：显示一个图像
    参数：
    image：图像数据（2维 or 3维数组）    
    返回值：NULL
    """

    def show_data(self, data):
        shape = data.shape

        # 如果是2维，就显示灰度图像
        if 2 == len(shape):
            plt.imshow(data, cmap='Greys_r')
        # 否则，就是3维，正常显示
        else:
            plt.imshow(data)

        plt.axis('off')  # 不显示坐标轴
        plt.show()


"""
test
"""


def test():
    file_name = "./dog1.bmp"
    image = MyImage()

    image.show_file(file_name)

    gray, err = image.gray_image(file_name)
    image.show_data(gray)
