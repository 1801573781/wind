"""
Function：Neural Network (base class)
Author：lzb
Date：2021.01.07
"""

import numpy.matlib
import numpy as np

import time

from gl import errorcode
from gl import common_function
from gl.array_string import array_2_string

from activation.last_hop_activation import DichotomyLHA
from loss.loss import MSELoss

"""
class：NeuralNetwork 神经网络(base class)

特别说明：
语义上的 vector，在代码中，实际上是一个 [n, 1] 的 matrix
"""


class NeuralNetwork:
    # 神经网络输入样本，向量维度
    _sx_dim = 0

    # 神经网络输出样本，向量维度
    _sy_dim = 0

    # 神经网络层数
    _layer_count = 0

    # 每一层神经元的数量
    _neuron_count_list = None

    # 每一层 w 参数，w 是个 matrix（BP 网络） or 3维数组（卷积网络）
    _W = None

    # 每一层 b 参数，b 是个 vector（BP 网络） or 2维数组（卷积网络）
    _B = None

    # 每一层 w 参数的 shape list（除了卷积网络，这个参数没有意义）
    _w_shape_list = None

    # 样本数量
    _sample_count = 0

    # 样本输入列表(Sample X list)，每一个输入样本是一个 vector
    _sx_list = None

    # 样本输出列表(Sample Y list)，每一个输出样本是一个 vector
    _sy_list = None

    # 循环训练的最大次数
    _loop_max = 1

    # 学习效率
    _rate = 0

    # 激活函数对象（class Activation 的实例）
    _activation = None

    # 最后一跳激活函数对象（class LastHopActivation 的实例）
    _last_hop_activation = DichotomyLHA()

    # 损失函数
    _loss = MSELoss()

    def __init__(self, activation, last_hop_activation=None, loss=None):
        """
        构造函数
        :param activation: 激活函数对象
        :param last_hop_activation: 后一跳激活函数对象
        :param loss: 损失函数对象
        """

        self._activation = activation

        if last_hop_activation is not None:
            self._last_hop_activation = last_hop_activation

        if loss is not None:
            self._loss = loss

    ''''''

    def train(self, sx_list, sy_list, loop_max, neuron_count_list, rate, w_shape_list=None):
        """
        功能：神经网络训练\n
        参数：\n
        sx_list：训练样本输入列表\n
        sy_list：训练样本输出列表\n
        loop_max：循环训练的最大次数 \n
        neuron_count_list：每一层神经元数量(对于卷积网络，这个参数没有意义)\n
        rate：学习效率 \n
        activation：激活函数对象\n
        last_hop_activation：最后一跳激活函数对象\n
        loss：损失函数对象\n
        w_shape_list：每一层 w 参数的 shape list（除了卷积网络，这个参数没有意义）\n
        返回值：错误码\n
        """

        # 1. 成员变量赋值
        self._sx_list = sx_list
        self._sy_list = sy_list
        self._loop_max = loop_max
        self._rate = rate

        # 如果是卷积网络，这个参数没有意义（如果是卷积网络，直接传入 None 即可）
        self._neuron_count_list = neuron_count_list

        # 如果不是卷积网络，这个参数，没有意义（如果不是卷积网络，直接传入默认值即可）
        self._w_shape_list = w_shape_list

        # 2. 校验
        err = self._valid()
        if errorcode.SUCCESS != err:
            print("\nvalid error, errcode = %d\n" % err)
            return err

        # 3. 初始化 W, B，及其他参数
        self._init_other_para()

        # 4. 训练
        return self._train()

    """
    功能：参数校验
    参数：NULL    
    返回值：错误码    
    """

    def _valid(self):
        # 1. 校验每层神经元
        self._valid_layer_neuron()

        # 2. 输入样本与输出样本
        err = self._valid_sample()
        if errorcode.SUCCESS != err:
            return err

        # 3. 最大循环训练次数，须 >= 1
        if 1 > self._loop_max:
            return errorcode.FAILED

        return errorcode.SUCCESS

    """
    功能：校验每层神经元
    参数：NULL    
    返回值：错误码    
    """

    def _valid_layer_neuron(self):
        # 1. 神经网络层数，须 >= 1
        layer_count = len(self._neuron_count_list)

        if 1 > layer_count:
            return errorcode.FAILED

        # 2. 每层的神经元个数，须 >= 1
        for layer in range(0, layer_count):
            count = self._neuron_count_list[layer]

            if 1 > count:
                return errorcode.FAILED

    """
    功能：校验样本
    参数：NULL    
    返回值：错误码    
    """

    def _valid_sample(self):
        # 1 输入样本的数量与输出样本的数量，须相同
        len1 = len(self._sx_list)
        len2 = len(self._sy_list)

        if len1 != len2:
            return errorcode.FAILED

        # 2 样本数量，须 >= 1
        sample_count = len(self._sx_list)
        if 1 > sample_count:
            return errorcode.FAILED

        # 3. 样本向量维度

        # 输入向量维度
        sx_dim = len(self._sx_list[0])

        # 输出向量维度
        layer_count = len(self._neuron_count_list)
        sy_dim = self._neuron_count_list[layer_count - 1]

        # 3.1 输入样本/输出样本，向量维度 > 1
        if (1 > sx_dim) or (1 > sy_dim):
            return errorcode.FAILED

        # 3.2 每一个输入/输出样本的向量维度
        for i in range(0, sample_count):
            shape_in = self._sx_list[i].shape
            shape_out = self._sy_list[i].shape

            # 输入样本的向量维度
            if shape_in[0] != sx_dim:
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

    """
    功能：初始化其它参数
    参数：NULL   
    返回值：错误码
    """

    def _init_other_para(self):
        # 每一层 w、B 参数，w 是个2维数组，b 是个2维数组
        self._W = list()
        self._B = list()

        # 样本数量
        self._sample_count = len(self._sx_list)

        # 神经网络输入，向量维度
        self._sx_dim = len(self._sx_list[0])

        # 神经网络的层数
        self._layer_count = len(self._neuron_count_list)

        # 神经网络输出，向量维度
        self._sy_dim = self._neuron_count_list[self._layer_count - 1]

        # 第1层 w 参数，w 是一个2维数组
        w = np.random.random((self._neuron_count_list[0], self._sx_dim))
        self._W.append(w)

        # 第2层~第layer-1层 w 参数，w 是一个2维数组
        for i in range(1, self._layer_count):
            w = np.random.random((self._neuron_count_list[i], self._neuron_count_list[i - 1]))
            self._W.append(w)

        # 第1层 ~ 第layer-1层 b 参数，b 是一个向量
        for i in range(0, self._layer_count):
            b = np.zeros([self._neuron_count_list[i], 1])
            self._B.append(b)

        return errorcode.SUCCESS

    """
    功能：训练（protected 函数）
    参数：NULL   
    返回值：错误码
    """

    def _train(self):
        # 循环训练次数
        loop = 0

        # 打印开始时间
        localtime = time.asctime(time.localtime(time.time()))
        print("\nbegin time = " + localtime + "\n")

        while 1:
            if loop >= self._loop_max:
                # 打印结束时间
                localtime = time.asctime(time.localtime(time.time()))
                print("\nend time = " + localtime + "\n")

                # 打印最后一轮参数
                self._print_w_b_loop(loop)

                break

            # 打印每一轮的参数
            # self._print_w_b_loop(loop)

            # 1. 每一轮训练之前，预准备工作
            self._pre_train()

            loop = loop + 1

            # 2. 训练每一个样本
            for i in range(0, self._sample_count):
                # 第 i 个训练样本
                sx = self._sx_list[i]
                sy = self._sy_list[i]

                # 2.1 第 m 个训练样本，经过（多层）神经网络的计算
                nn_y_list = self._calc_nn(sx)

                # 2.2 最后一跳修正
                nn_y = nn_y_list[len(nn_y_list) - 1]
                last_hop_y = self._last_hop_activation.train_activation(nn_y)
                nn_y_list.append(last_hop_y)

                # 2.3 根据计算结果，修正参数 W，B
                self._modify_wb(nn_y_list, sx, sy)

        return errorcode.SUCCESS

    # 每一轮训练之前预准备工作
    def _pre_train(self):
        """
        每一轮训练之前预准备工作（一般来说，啥都不用做）
        :return: NULL
        """
        pass

    # 计算整个网络的输出
    def _calc_nn(self, sx):
        """
        计算整个网络的输出
        :param sx: 神经网络的输入
        :return: 整个神经网络，每一层的输出
        """

        x = sx
        y = 0
        nn_y_list = list()

        # 逐层计算
        for layer in range(0, self._layer_count):
            # 计算该层的输出
            y = self._calc_layer(x, layer)

            # 将该层的输出，记录下来
            nn_y_list.append(y)

            # 本层输出，等于下一层的输入
            x = y

        # 返回逐层计算的结果
        return nn_y_list

    ''''''

    def _calc_layer(self, x, layer):
        """
        计算神经网络某一层的输出
        :param x: 该层神经网络的输入，x 是一个向量
        :param layer: 当前的层数
        :return: y，该层神经网络的输出， y 是一个向量
        """

        # 获取该层的参数：w, b
        w = self._W[layer]
        b = self._B[layer]

        y = np.matmul(w, x) + b

        y = y + self._calc_recurrent(layer)

        # 针对每一个元素，调用激活函数
        row = len(y)

        for i in range(0, row):
            y[i, 0] = self._activation.active(y[i, 0])

        return y

    ''''''

    def _calc_recurrent(self, layer):
        """
        计算循环神经网络， u * h(t - 1) ，默认值是 0
        :param layer: 层数
        :return: u * h(t - 1)
        """
        return 0

    ''''''

    def _modify_wb(self, nn_y_list, sx, sy):
        """
        功能：修正 W，B
        参数：
        nn_y_list：神经网路计算的每一层结果，nn_y 是一个向量
        sx：训练样本的输入，sx 是一个向量
        sy：训练样本的输出，sy 是一个向量
        返回值：NULL
        """
        pass

    ''''''

    def predict(self, sx_list, sy_list):
        """
        功能：预测
        参数：
        sx_list：待预测的样本列表，其中 sx 是向量
        返回值：预测结果
        """

        count = len(sx_list)
        py_list = list()

        for i in range(0, count):
            sx = sx_list[i]
            nn_y_list = self._calc_nn(sx)

            # 最后一层的 nn_y，才是神经网络的最终输出
            nn_y = nn_y_list[len(nn_y_list) - 1]

            # 修正一下
            last_hop_y = self._last_hop_activation.predict_activation(nn_y)

            # 然后再添加到预测列表
            py_list.append(last_hop_y)

        return py_list

    """
    功能：打印 W, B, loop
    参数：
    loop：神经网络的训练次数
    返回值：NULL       
    """

    def _print_w_b_loop(self, loop):
        print("\n")
        print("训练次数 = %d\n" % loop)

        for layer in range(0, self._layer_count):
            print("层数 ＝ %d" % layer)

            print("W:")
            # print(self.W[layer])
            print(array_2_string(self._W[layer]))

            print("\nB:")
            # print(self.B[layer])
            print(array_2_string(self._B[layer]))

            if layer < self._layer_count - 1:
                print("\n")

    """
    功能：桩函数，设置参数（不必训练，直接设置）
    参数：    
    sx_dim：神经网络输入，向量维度    
    layer_count：神经网络层数
    neuron_count_list：每一层神经元的数量(Neuron Count)    
    W：每一层 w 参数 列表，w 是个 matrix    
    B：每一层 b 参数 列表，b 是个 vector
    返回值：NULL
    """

    def stub_set_para(self, sx_dim, neuron_count_list, W, B, activation):
        # 神经网络输入，向量维度
        self._sx_dim = sx_dim

        # 每一层神经元的数量(Neuron Count)
        self._neuron_count_list = neuron_count_list

        # 神经网络层数
        self._layer_count = len(W)

        # 每一层 w 参数，w 是个 matrix
        self._W = W

        # 每一层 b 参数，b 是个 vector
        self._B = B

        # 激活函数对象
        self._activation = activation
