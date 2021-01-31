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

from activation.last_hop_activation import LastHopActivation, DichotomyLHA, SoftMaxLHA

"""
class：NeuralNetwork 神经网络(base class)

特别说明：
语义上的 vector，在代码中，实际上是一个 [n, 1] 的 matrix
"""


class NeuralNetwork:
    # 神经网络输入样本，向量维度
    sx_dim = 0

    # 神经网络输出样本，向量维度
    sy_dim = 0

    # 神经网络层数
    layer_count = 0

    # 每一层神经元的数量
    neuron_count_list = None

    # 每一层 w 参数，w 是个 matrix（BP 网络） or 3维数组（卷积网络）
    W = None

    # 每一层 b 参数，b 是个 vector（BP 网络） or 2维数组（卷积网络）
    B = None

    # 每一层 w 参数的 shape list（除了卷积网络，这个参数没有意义）
    w_shape_list = None

    # 样本数量
    sample_count = 0

    # 样本输入列表(Sample X list)，每一个输入样本是一个 vector
    sx_list = None

    # 样本输出列表(Sample Y list)，每一个输出样本是一个 vector
    sy_list = None

    # 循环训练的最大次数
    loop_max = 1

    # 学习效率
    rate = 0

    # 激活函数对象（class Activation 的实例）
    activation = None

    # 最后一跳激活函数对象（class LastHopActivation 的实例）
    last_hop_activation = DichotomyLHA()

    """
    功能：训练（public 函数）
    参数：
    sx_list：训练样本输入列表
    sy_list：训练样本输出列表
    loop_max：循环训练的最大次数    
    neuron_count_list：每一层神经元数量(对于卷积网络，这个参数没有意义)
    rate：学习效率    
    activation：激活函数对象    
    w_shape_list：每一层 w 参数的 shape list（除了卷积网络，这个参数没有意义）       
    返回值：错误码
    """

    def train(self, sx_list, sy_list, loop_max, neuron_count_list, rate,
              activation, last_hop_activation=None, w_shape_list=None):
        # 1. 成员变量赋值
        self.sx_list = sx_list
        self.sy_list = sy_list
        self.loop_max = loop_max
        self.rate = rate
        self.activation = activation

        if last_hop_activation is not None:
            self.last_hop_activation = last_hop_activation

        # 如果是卷积网络，这个参数没有意义（如果是卷积网络，直接传入 None 即可）
        self.neuron_count_list = neuron_count_list

        # 如果不是卷积网络，这个参数，没有意义（如果不是卷积网络，直接传入默认值即可）
        self.w_shape_list = w_shape_list

        # 2. 校验
        err = self._valid()
        if errorcode.SUCCESS != err:
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
        if 1 > self.loop_max:
            return errorcode.FAILED

        return errorcode.SUCCESS

    """
    功能：校验每层神经元
    参数：NULL    
    返回值：错误码    
    """
    def _valid_layer_neuron(self):
        # 1. 神经网络层数，须 >= 1
        layer_count = len(self.neuron_count_list)

        if 1 > layer_count:
            return errorcode.FAILED

        # 2. 每层的神经元个数，须 >= 1
        for layer in range(0, layer_count):
            count = self.neuron_count_list[layer]

            if 1 > count:
                return errorcode.FAILED

    """
    功能：校验样本
    参数：NULL    
    返回值：错误码    
    """

    def _valid_sample(self):
        # 1 输入样本的数量与输出样本的数量，须相同
        len1 = len(self.sx_list)
        len2 = len(self.sy_list)

        if len1 != len2:
            return errorcode.FAILED

        # 2 样本数量，须 >= 1
        sample_count = len(self.sx_list)
        if 1 > sample_count:
            return errorcode.FAILED

        # 3. 样本向量维度

        # 输入向量维度
        sx_dim = len(self.sx_list[0])

        # 输出向量维度
        layer_count = len(self.neuron_count_list)
        sy_dim = self.neuron_count_list[layer_count - 1]

        # 3.1 输入样本/输出样本，向量维度 > 1
        if (1 > sx_dim) or (1 > sy_dim):
            return errorcode.FAILED

        # 3.2 每一个输入/输出样本的向量维度
        for i in range(0, sample_count):
            shape_in = self.sx_list[i].shape
            shape_out = self.sy_list[i].shape

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
        self.W = list()
        self.B = list()

        # 样本数量
        self.sample_count = len(self.sx_list)

        # 神经网络输入，向量维度
        self.sx_dim = len(self.sx_list[0])

        # 神经网络的层数
        self.layer_count = len(self.neuron_count_list)

        # 神经网络输出，向量维度
        self.sy_dim = self.neuron_count_list[self.layer_count - 1]

        # 第1层 w 参数，w 是一个矩阵
        w = np.matlib.rand(self.neuron_count_list[0], self.sx_dim)
        # w = np.random.random((self.neuron_count_list[0], self.sx_dim))
        self.W.append(w)

        # 第2层~第layer-1层 w 参数，w 是一个矩阵
        for i in range(1, self.layer_count):
            w = np.matlib.rand(self.neuron_count_list[i], self.neuron_count_list[i - 1])
            # w = np.random.random((self.neuron_count_list[i], self.neuron_count_list[i - 1]))
            self.W.append(w)

        # 第1层 ~ 第layer-1层 b 参数，b 是一个向量
        for i in range(0, self.layer_count):
            b = np.zeros([self.neuron_count_list[i], 1])
            self.B.append(b)

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
            if loop >= self.loop_max:
                # 打印结束时间
                localtime = time.asctime(time.localtime(time.time()))
                print("\nend time = " + localtime + "\n")

                # 打印参数
                self._print_w_b_loop(loop)

                break

            # 1. 打印每一轮的参数
            # self._print_w_b_loop(loop)

            loop = loop + 1

            # 2. 训练每一个样本
            for i in range(0, self.sample_count):
                # 第 i 个训练样本
                sx = self.sx_list[i]
                sy = self.sy_list[i]

                # 2.1 第 m 个训练样本，经过（多层）神经网络的计算
                nn_y_list = self._calc_nn(sx)

                # 2.2 最后一跳修正
                nn_y = nn_y_list[len(nn_y_list) - 1]
                self.last_hop_activation.train_activation(nn_y)

                # 2.3 根据计算结果，修正参数 W，B
                self._modify_wb(nn_y_list, sx, sy)

        return errorcode.SUCCESS

    """
    功能：计算整个网络的输出
    参数：
    s：神经网络的输入    
    返回值：整个神经网络，每一层的输出
    """

    def _calc_nn(self, sx):
        x = sx
        y = 0
        nn_y_list = list()

        # 逐层计算
        for layer in range(0, self.layer_count):
            # 计算该层的输出
            y = self._calc_layer(x, layer)

            # 将该层的输出，记录下来
            nn_y_list.append(y)

            # 本层输出，等于下一层的输入
            x = y

        # 返回逐层计算的结果
        return nn_y_list

    """
    功能：计算某一层神经网络的输出
    参数：
    x：该层神经网络的输入，x 是一个向量
    w: 该层神经网络的 w 参数, w 是一个矩阵
    b：该层神经网络的 b 参数，b 是一个向量
    返回值：y，该层神经网络的输出（sigmoid(w * x + b)）， y 是一个向量
    """

    def _calc_layer(self, x, layer):
        # 获取该层的参数：w, b
        w = self.W[layer]
        b = self.B[layer]

        y = np.matmul(w, x) + b

        # 针对每一个元素，调用激活函数
        row = len(y)

        for i in range(0, row):
            y[i, 0] = self.activation.active(y[i, 0])

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

            # 修正一下
            self.last_hop_activation.predict_activation(nn_y)
            """
            for j in range(0, self.sy_dim):
                nn_y[j, 0] = self.activation.revise(nn_y[j, 0])
            """

            # 然后再添加到预测列表
            py_list.append(nn_y)

        accuracy = common_function.calculate_accuracy(py_list, sy_list)

        print("\n")
        print("accuracy = %f\n" % accuracy)

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

        for layer in range(0, self.layer_count):
            print("层数 ＝ %d" % layer)

            print("W:")
            # print(self.W[layer])
            print(array_2_string(self.W[layer]))

            print("\nB:")
            # print(self.B[layer])
            print(array_2_string(self.B[layer]))

            if layer < self.layer_count - 1:
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
        self.sx_dim = sx_dim

        # 每一层神经元的数量(Neuron Count)
        self.neuron_count_list = neuron_count_list

        # 神经网络层数
        self.layer_count = len(W)

        # 每一层 w 参数，w 是个 matrix
        self.W = W

        # 每一层 b 参数，b 是个 vector
        self.B = B

        # 激活函数对象
        self.activation = activation


