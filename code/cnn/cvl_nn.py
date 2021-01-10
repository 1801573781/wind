"""
Function：Convolution Neural Network
Author：lzb
Date：2021.01.10
"""

import numpy as np

from nn.neural_network import NeuralNetwork
from gl import errorcode


"""
class：CVLNeuralNetwork，卷积神经网络
说明：
1、继承自 NeuralNetwork
2、重载 _modify_wb 函数

特别说明：
1、每一层的 w，还是一个 matrix，不过它的定义是：卷积核
2、每一层的神经元个数，将由上一层的神经元个数和卷积参数确定（w，s，padding），而不是由外部输入
3、卷积参数：步长 s = 1，补零 padding = 0
4、激活函数是 ReLU 函数
5、输入样本 sx，为了计算方便，将是1个3维矩阵，为了方便理解，可以这么认为：
A. 第1维：图像像素的 x 坐标
B. 第2维：图像像素的 y 坐标
C. 第3维：图像像素的颜色值。如果是 RGB 图像，则第3维有3个值（r, g, b），如果是灰度图像，则第3维只有1个值（gray）
6、从这个意义上讲，参数 sx_dim 将变为1个”3元祖“：sx_dim[0] 表示图像的宽度，sx_dim[1] 表示图像的高度，sx_dim[2] 表示图像的颜色深度
7、输出样本，暂时还定义为分类，所以它还是一个1维向量。因此，sy_dim 还是表示1维向量的元素个数（1维向量的维度）
"""


class CVLNeuralNetwork(NeuralNetwork):
    # 颜色深度（颜色维度）
    # 这个成员变量命名为 color_depth，那就意味着这个 class 只是处理图像相关了
    # 这样的命名，坏处是：与图像处理太过耦合，好处是：易于理解（尤其是暂时不知道卷积网络还能做什么的情况下）
    color_depth = 1

    """
    功能：校验样本
    参数：NULL    
    返回值：错误码
    说明：理解卷积神经网络，理解卷积对图像的处理，恐怕是从样本的校验开始
    """

    def _valid_sample(self):
        # 1 输入样本的数量与输出样本的数量，须相同（equal with parent class）
        len1 = len(self.sx_list)
        len2 = len(self.sy_list)

        if len1 != len2:
            return errorcode.FAILED

        # 2 样本数量，须 >= 1（equal with parent class）
        sample_count = len(self.sx_list)
        if 1 > sample_count:
            return errorcode.FAILED

        # 3. 样本向量维度（different with parent class）

        # 3.1 输入向量维度
        sx_dim = self.sx_list[0].shape

        # 输入向量必须是3维矩阵：图像的宽度、高度，颜色的深度
        if 3 != sx_dim:
            return errorcode.FAILED

        # 如果颜色深度大于3 or 小于1，则非法
        if (3 < sx_dim[2]) or (1 > sx_dim[2]):
            return errorcode.FAILED

        # 3.2 输出向量
        sy_dim = len(self.sy_list[0])

        # 输出样本向量维度 > 1
        if 1 > sy_dim:
            return errorcode.FAILED

        # 3.2 每一个输入/输出样本的向量维度
        for i in range(0, sample_count):
            shape_in = self.sx_list[i].shape
            shape_out = self.sy_list[i].shape

            # 输入样本的维度须相等
            if 0 != cmp(sx_dim, shape_in):
                return errorcode.FAILED

            # 输入样本只能有1列（因为是个向量）
            if shape_in[1] != 1:
                return errorcode.FAILED

            # 输出样本的向量维度
            if shape_out[0] != sy_dim:
                return errorcode.FAILED

            # 输出样本只能有1列（因为是个向量）
            if shape_out[1] != 1:
                return errorcode.FAILED

        return errorcode.SUCCESS

