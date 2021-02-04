"""
Function：构建猫、狗的样本（灰度图）
Author：lzb
Date：2020.12.25
"""

import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np

import os

from gl.common_enum import ArrayDim
from my_image import my_image
from my_image.my_image import gray_file
from sample.fully_connected_sample import FullConnectedSample


class ImageSoftMaxSample(FullConnectedSample):
    """
    1、暂时只使用 0~9 图像，进行图像识别\n
    2、图像的目录，暂时写死\n
    """

    def create_sample(self, sx_dim, sy_dim):
        """
        功能：创建样本\n
        参数：\n
        sx_dim：样本，输入向量的维度\n
        sy_dim：样本，输出向量的维度\n
        返回值：NULL\n
        """

        # 1. 初始化
        self.sx_dim = sx_dim
        self.sy_dim = sy_dim

        self.sx_list = list()
        self.sy_list = list()

        image_path = "./../picture/number3"

        # 2. 构建 sx_list, sy_list

        # 2.1 获取 image_path 下所有文件
        for root, dirs, files in os.walk(image_path, topdown=False):
            for name in files:
                image_file_name = os.path.join(root, name)
                # 2.2 构建 sx
                sx = self._create_sx(image_file_name)
                self.sx_list.append(sx)

                # 2.3 构建 sy
                sy = self._create_sy(image_file_name)
                self.sy_list.append(sy)

    ''''''

    @staticmethod
    def _create_sx(image_file_name):
        """
        功能：将一个图像文件转换为训练样本的输入 \n
        参数：\n
        image_file_name：图像文件名 \n
        返回值：sx \n
        """

        # 取图像灰度值
        gray, err = gray_file(image_file_name, ArrayDim.THREE)

        # 将图像数据中的0转换为极小值
        my_image.array_0_tiny(gray)

        # 归一化
        gray = my_image.normalize(gray, my_image.NormalizationType.NORMAL)

        # 将灰度图像值，从3维数组转变为1维数组（list）
        gray = my_image.array_3_1(gray)

        return gray

    ''''''

    @staticmethod
    def _create_sy(image_file_name):
        """
        通过解析图像文件名，构建为训练样本的输出
        :param image_file_name: 图像文件名
        :return: 训练样本的输出
        """

        # sy 是一个 10 维向量
        sy = np.zeros([10, 1])

        # 判断文件名中所包含的数字
        for i in range(0, 10):
            if str(i) in image_file_name:
                sy[i][0] = 1
                break

        return sy

