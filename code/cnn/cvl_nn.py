"""
Function：Convolution Neural Network
Author：lzb
Date：2021.01.10
"""

import numpy as np
import operator

from nn.neural_network import NeuralNetwork
from gl import errorcode

from cnn.convolution import Convolution, ConvolutionType, Reversal, cal_cvl_wh

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
    # 可以理解为图像宽度
    width = 1

    # 可以理解为图像高度
    height = 1

    # 可以理解为颜色深度（颜色维度）
    depth = 1

    # 每一层神经网络的输出（经过激活函数以后的输出），a 是一个三维数组
    a_list = None

    # 卷积步长
    s = 1

    # 卷积两端补齐长度
    padding = 0

    # 卷积类型
    cvl_type = ConvolutionType.Narrow

    # 是否翻转卷积
    rev = Reversal.NO_REV

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

        # 2 样本数量，须 >= 1（same as parent class）
        sample_count = len(self.sx_list)
        if 1 > sample_count:
            return errorcode.FAILED

        # 3. 样本向量维度（different with parent class）

        # 3.1 输入向量维度
        sx_dim = self.sx_list[0].shape

        # 输入向量必须是3维矩阵：图像的宽度、高度，颜色的深度
        if 3 != sx_dim:
            return errorcode.FAILED

        width = sx_dim[0]
        height = sx_dim[1]
        depth = sx_dim[2]

        # 图像宽度/高度须大于0
        if (0 > width) or (0 > height):
            return errorcode.FAILED

        # 颜色深度须介于1~3之间
        if (3 < depth) or (1 > depth):
            return errorcode.FAILED

        # 3.2 输出向量
        sy_dim = len(self.sy_list[0])

        # 输出样本向量维度 > 1
        if 1 > sy_dim:
            return errorcode.FAILED

        # 3.2 每一个输入/输出样本的维度
        for i in range(0, sample_count):
            shape_in = self.sx_list[i].shape
            shape_out = self.sy_list[i].shape

            # 输入样本的维度须相等(都是3维)
            if not operator.eq(sx_dim, shape_in):
                return errorcode.FAILED

            # 输入样本的宽度、高度、深度
            if (shape_in[0] != width) or (shape_in[1] != height) or (shape_in[2] != depth):
                return errorcode.FAILED

            # 输出样本的向量维度
            if shape_out[0] != sy_dim:
                return errorcode.FAILED

            # 输出样本只能有1列（因为是个向量）
            if shape_out[1] != 1:
                return errorcode.FAILED

        return errorcode.SUCCESS

    """
    功能：初始化其它参数
    参数：NULL   
    返回值：错误码
    """

    def _init_other_para(self):
        # 样本数量
        self.sample_count = len(self.sx_list)

        # 神经网络输入，维度(3维)
        self.sx_dim = self.sx_list[0].shape

        # 图像宽度，高度，深度
        self.width = self.sx_dim[0]
        self.height = self.sx_dim[1]
        self.depth = self.sx_dim[2]

        # 神经网络输出，向量维度
        self.sy_dim = self.neuron_count_list[self.layer_count - 1]

        # 初始化每一层神经网络的输出
        # self._init_cnn_a()

        # 初始化 self.layer_count, self.B

    """
    功能：初始化每一层神经网络的输出（经过激活函数以后的输出）
    参数：NULL
    返回值：NULL
    """

    """
    def _init_cnn_a(self):
        # a_list 是一个列表，a 是一个3维数组
        self.a_list = list()

        # 针对每一层进行初始化
        x = 0
        y = 0
        for layer in range(0, self.layer_count):
            # 如果是第一层，x 就是样本输入
            if 0 == layer:
                x = self.sx_list[0]
            # 否则的话，x 是上一层的输入
            else:
                x = y

            # 每一层的卷积核
            w = self.W[layer]

            width, height = cal_cvl_wh(w, x, self.s)

            # 每一层的输出，都是 [width, height, 1] 3维数组
            a = np.zeros([width, height, 1])

            self.a_list.append(a)
    """

    """
    功能：计算某一层神经网络的输出
    参数：
    x：该层神经网络的输入，x 是一个3维数组
    w: 该层神经网络的 w 参数, w 是一个3维数组
    b：该层神经网络的 b 参数，b 是一个2维数组
    返回值：y，该层神经网络的输出（sigmoid(cvl(w, x) + b)）， y 是一个3维数字
    """

    def _calc_layer(self, x, layer):
        # 1、获取该层的参数：w, b
        w = self.W[layer]
        b = self.B[layer]

        # 2、构建卷积对象
        cvl = Convolution()

        # 3、计算卷积结果
        y = cvl.convolution_sum_depth(w, x)

        # 4. y = y + b
        y_width = y.shape[0]
        y_height = y.shape[1]

        for i in range(0, y_width):
            for j in range(0, y_height):
                y[i, j, 0] = b[i, j]  # y 的 depth 肯定是1

        # 针对每一个元素，调用激活函数
        for i in range(0, y_width):
            for j in range(0, y_height):
                y[i, j, 0] = self.activation.active(y[i, j, 0])  # y 的 depth 肯定是1

        return y

    """
    功能：修正 W，B
    参数：
    nn_y_list：神经网路计算的每一层结果，nn_y 是一个向量
    sx：训练样本的输入，sx 是一个向量
    sy：训练样本的输出，sy 是一个向量 
    返回值：NULL
    """

    def _modify_wb(self, nn_y_list, sx, sy):
        pass
