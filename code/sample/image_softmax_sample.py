"""
Function：构建猫、狗的样本（灰度图）
Author：lzb
Date：2020.12.25
"""

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

    def create_sample(self, image_root_path, confuse=True):
        """
        功能：创建样本\n
        参数：\n
        sx_dim：样本，输入向量的维度\n
        sy_dim：样本，输出向量的维度\n
        返回值：NULL\n
        """

        # 1. 初始化
        self._sx_list = list()
        self._sy_list = list()

        # 2. 构建 sx_list, sy_list

        # 获取 image_path 下所有文件
        for root, dirs, files in os.walk(image_root_path, topdown=True):
            group_count = len(dirs)

            if 0 == group_count:
                continue
            else:
                for directory in dirs:
                    image_file_path = os.path.join(root, directory)
                    index = int(directory)  # directory 以 数字命名
                    # 创建样本
                    self._create_sample(image_file_path, index)

                # 样本混淆
                if confuse:
                    self._confuse(group_count)

                return  # 直接 return，不再继续创建样本

    ''''''

    def _create_sample(self, image_file_path, index):
        """
        创建样本
        :param image_file_path:图像文件路径
        :param index:图像文件路径所对应的数字
        :return:NULL
        """

        # 获取 image_file_path 下所有文件
        for root, dirs, files in os.walk(image_file_path, topdown=False):
            for name in files:
                image_file_name = os.path.join(root, name)
                # 1 构建 sx
                sx = self._create_sx(image_file_name)
                self._sx_list.append(sx)

                # 2 构建 sy
                sy = self._create_sy(index)
                self._sy_list.append(sy)

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

        gray = gray / 40

        return gray

    ''''''

    @staticmethod
    def _create_sy(index):
        """
        通过解析图像文件名，构建为训练样本的输出
        :return: 训练样本的输出
        """

        # sy 是一个 10 维向量
        sy = np.zeros([10, 1])

        sy[index][0] = 1

        return sy

    ''''''

    def _confuse(self, group_count):
        """
        将训练样本的顺序混淆一下
        :return:NULL
        """

        sx_list = list()
        sy_list = list()

        count = int(len(self._sx_list) / group_count)

        for i in range(0, count):
            for j in range(0, group_count):
                index = j * count + i
                sx_list.append(self._sx_list[index])
                sy_list.append(self._sy_list[index])

        self._sx_list = sx_list
        self._sy_list = sy_list

    ''''''

    def create_sample_ex(self, count):
        """
        认为构建样本，样本的可区分度强
        :return:NULL
        """

        self._sx_list = list()
        self._sy_list = list()

        for i in range(0, count):
            sx_0 = 0.5 * np.random.random((400, 1))
            sx_0 = sx_0 / 40
            self._sx_list.append(sx_0)

            sy_0 = np.zeros([10, 1])
            sy_0[0][0] = 1
            self._sy_list.append(sy_0)

            sx_1 = 0.5 * np.random.random((400, 1)) + 0.5
            sx_1 = sx_1 / 40
            self._sx_list.append(sx_1)

            sy_1 = np.zeros([10, 1])
            sy_1[1][0] = 1
            self._sy_list.append(sy_1)
