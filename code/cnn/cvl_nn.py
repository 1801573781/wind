"""
Function：Convolution Neural Network
Author：lzb
Date：2021.01.10
"""

import numpy as np
import operator

from nn.neural_network import NeuralNetwork
from gl import errorcode
from gl.common_enum import ArrayDim
from gl.common_function import rand_array_3

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

    # 卷积对象
    cvl = None

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
    功能：构造函数
    参数：
    cvl：卷积对象
    返回值：NULL
    """

    def __init__(self, cvl):
        self.cvl = cvl

    """
    功能：参数校验
    参数：NULL    
    返回值：错误码    
    """

    def _valid(self):
        # 调用父类的 _valid
        err = super()._valid()

        if errorcode.SUCCESS != err:
            return err

        # 校验 w_shape_list

        if self.w_shape_list is None:
            return errorcode.FAILED

        if 0 >= len(self.w_shape_list):
            return errorcode

        # 这里只处理3维数组
        shape = self.w_shape_list[0]

        if 3 != len(shape):
            return errorcode.FAILED

        if (0 >= shape[0]) or (0 >= shape[1]) or (0 >= shape[1]):
            return errorcode.FAILED

        return errorcode.SUCCESS

    """
    功能：校验每层神经元
    参数：NULL    
    返回值：错误码
    说明：对于卷积神经网络来说，这里不需要校验
    """

    def _valid_layer_neuron(self):
        return errorcode.SUCCESS

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

        # 3. 样本数组维度（different with parent class）

        # 3.1 输入数组维度
        sx_dim = self.sx_list[0].shape

        # 输入样本必须是3维样本：图像的宽度、高度，颜色的深度
        if ArrayDim.THREE.value != len(sx_dim):
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

            """
            # 输出样本的向量维度
            if shape_out[0] != sy_dim:
                return errorcode.FAILED

            # 输出样本只能有1列（因为是个向量）
            if shape_out[1] != 1:
                return errorcode.FAILED
            """

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
        # self.sy_dim = self.neuron_count_list[self.layer_count - 1]

        # 初始化 self.layer_count
        self.layer_count = len(self.w_shape_list)

        # 初始化 W, B
        self._init_w_b()

        return errorcode.SUCCESS

    """
    功能：初始化 W, B
    参数：NULL
    返回值：NULL
    
    特别说明，这里就假设 w 是 3维数组
    """

    def _init_w_b(self):
        # 1. W，B 是 list
        self.W = list()
        self.B = list()

        # 2. 针对每一层进行初始化
        b = 0
        for layer in range(0, self.layer_count):
            # 2.1 每一层的卷积核
            width = self.w_shape_list[layer][0]
            height = self.w_shape_list[layer][1]

            # 如果是第一层，depth = 输入层的 depth
            if 0 == layer:
                depth = self.w_shape_list[layer][2]
            # 否则的话，depth = 1
            else:
                depth = 1

            w = rand_array_3(width, height, depth)
            # w = np.zeros([width, height, depth])
            self.W.append(w)

            # 2.2 每一层的 b

            # 如果是第一层，x 就是样本输入
            if 0 == layer:
                x = self.sx_list[0]
            # 否则的话，x 是上一层的输出
            # 上一层的输出的 width，height 等同于 b
            else:
                x = b

            width, height = cal_cvl_wh(w, x, self.s)

            # 每一层的b，都是 [width, height, depth] 3维数组
            depth = 1  # b 的 depth = 1
            b = rand_array_3(width, height, depth)
            # b = np.zeros([width, height, depth])

            self.B.append(b)

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

        # 2、计算卷积结果
        y, err = self.cvl.convolution_sum_depth(w, x)
        # y, err = cvl.convolution(w, x)

        # 3. y = y + b
        y_width = y.shape[0]
        y_height = y.shape[1]
        y_depth = y.shape[2]

        for i in range(0, y_width):
            for j in range(0, y_height):
                for k in range(0, y_depth):
                    y[i, j, k] += b[i, j, k]

        # 针对每一个元素，调用激活函数
        for i in range(0, y_width):
            for j in range(0, y_height):
                for k in range(0, y_depth):
                    y[i, j, k] = self.activation.active(y[i, j, k])

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
        # 1. 后向传播，计算 ksi_list
        ksi_list = self.__bp(nn_y_list, sy)

        # 2. 通过 ksi_list，修正 W，B
        self.__modify_wb_by_ksi_list(ksi_list, sx, nn_y_list)

    """
    功能：后向传播，计算 ksi_list
    参数：
    nn_y_list：神经网路计算的每一层结果，nn_y 是一个3维数组    
    sy：训练样本的输出，sy 是一个3维数组
    返回值：ksi_list
    说明：
    1、ksi(代表希腊字母，音：科赛)，是一个3维数组，每层都有，代表目标函数 E 对每一层中间输出的偏导
    2、ksi_list 记录每一层的 ksi
    """

    def __bp(self, nn_y_list, sy):
        # 1. 初始化 ksi_list
        ksi_list = [0] * self.layer_count

        # 2. 计算最后一层 ksi

        # 2.1 计算误差(err)：最后一层的计算结果与样本输出结果的比较（计算结果 - 训练样本的输出）
        nn_y_last = nn_y_list[self.layer_count - 1]
        err = np.subtract(nn_y_last, sy)  # 不知道3维数组是否可以这样相减

        # 2.2 计算最后一层 ksi

        # 最后一层 ksi：ksi_last，ksi_last 是个[width, height, 1] 3维数组
        width = nn_y_last.shape[0]
        height = nn_y_last.shape[1]
        depth = nn_y_last.shape[2]  # 实际的值，depth = 1

        ksi_last = np.zeros([width, height, depth])

        # 计算 ksi_last 每一个元素
        for k in range(0, depth):
            for i in range(0, width):
                for j in range(0, height):
                    ksi_last[i, j, k] = err[i, j, k] * self.activation.derivative(nn_y_last[i, j, k])

        # 将 ksi_last 放置入 ksi_list
        ksi_list[self.layer_count - 1] = ksi_last

        # 3. 反向传播，计算：倒数第2层 ~ 第1层的 ksi
        for layer in range(self.layer_count - 2, -1, -1):
            # 下一层的 ksi
            ksi_next = ksi_list[layer + 1]

            # 下一层的 w
            w = self.W[layer + 1]

            # 当前层的 ksi
            ksi_cur, err = self.cvl.convolution(w, ksi_next, Reversal.REV, ConvolutionType.Wide)

            # 将当前层计算出的 ksi 放置到 ksiList
            ksi_list[layer] = ksi_cur

        # return 计算结果
        return ksi_list

    """
    功能：修正 W，B
    参数： 
    ksi_list：每一层的 ksi 的列表，ksi 是一个3维数组
    sx：输入样本，sx 是一个3维数组
    nn_y_list：神经网络的每一层的计算结果列表，nn_y 是一个3维数组    
    返回值：NULL  
    """

    def __modify_wb_by_ksi_list(self, ksi_list, sx, nn_y_list):
        # 逐层修正
        for layer in range(0, self.layer_count):
            # 当前层 w, b, ksi
            w = self.W[layer]
            b = self.B[layer]
            ksi = ksi_list[layer]

            # 上一层的输入
            if 0 == layer:
                v = sx
            else:
                v = nn_y_list[layer - 1]

            # 损失函数针对当前层的 w 的偏导(partial derivative)，w_pd 是1个3维数组
            w_pd, err = self.cvl.convolution(ksi, v)

            # 修正当前层的 w
            self.W[layer] = np.subtract(w, self.rate * w_pd)  # 不知道3维数组是否可以这样相减

            # 损失函数针对当前层的 b 的偏导(partial derivative)，b_pd 等于 ksi
            b_pd = ksi

            # 修正当前层的 b
            self.B[layer] = np.subtract(b, self.rate * b_pd)  # 不知道3维数组是否可以这样相减

    """
    功能：预测
    参数：
    sx_list：待预测的样本列表，其中 sx 是向量 
    返回值：预测结果
    """

    def predict(self, sx_list, sy_list):
        count = len(sx_list)
        py_list = list()

        for i in range(0, count):
            sx = sx_list[i]
            nn_y_list = self._calc_nn(sx)

            # 最后一层的 nn_y，才是神经网络的最终输出
            nn_y = nn_y_list[len(nn_y_list) - 1]

            # 然后再添加到预测列表
            py_list.append(nn_y)

        return py_list
