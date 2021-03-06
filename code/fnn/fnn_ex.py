"""
Function：分组训练的 FNN
Author：lzb
Date：2021.02.22

特别说明：
1、以前写的代码，class FNN，没有考虑分组训练
2、现在考虑分组训练，暂时先重写代码
3、以前的代码（FNN） 暂时先保留，待 class FNNEx 完善以后，再删除原先的代码
"""

import os
import pickle

import numpy as np

from gl import errorcode
from gl.array_string import array_2_string
from gl.common_function import get_local_time, unserialize_train_para

from activation.normal_activation import Sigmoid
from activation.last_hop_activation import DichotomyLHA
from loss.loss import MSELoss


class FnnEx:
    """
    分组训练的 FNN
    """

    # 神经网络输入样本，向量维度
    _sx_dim = 0

    # 神经网络输出样本，向量维度
    _sy_dim = 0

    # 神经网络层数
    _layer_count = 0

    # 每一层神经元的数量
    _neuron_count_list = None

    # 每一层 w 参数
    _w_layer = None

    # 每一层 w 参数的 delta
    _delta_w_layer = None

    # 每一层 b 参数
    _b_layer = None

    # 每一层 b 参数的 delta
    _delta_b_layer = None

    # 每一层 w 参数的 shape list（除了卷积网络，这个参数没有意义）
    _w_shape_layer = None

    # 样本数量
    _sample_count = 0

    # 训练样本分组列表(输入)
    _sx_group_list = None

    # 训练样本分组列表(输出)
    _sy_group_list = None

    # 循环训练的最大次数
    _loop_max = 1

    # 学习效率
    _rate = 0

    # 激活函数对象（class Activation 的实例）
    _activation = Sigmoid()

    # 最后一跳激活函数对象（class LastHopActivation 的实例）
    _last_hop_activation = DichotomyLHA()

    # 损失函数
    _loss = MSELoss()

    # 记录训练参数文件路径
    _para_file_path = 'gl/train_para/'

    # 是否通过反序列化初始化训练参数
    _init_from_unserialization = False

    # 训练参数初始化时所乘以的系数
    _alpha_para = 1

    def __init__(self, activation=None, last_hop_activation=None, loss=None):
        """
        构造函数
        :param activation: 激活函数对象
        :param last_hop_activation: 后一跳激活函数对象
        :param loss: 损失函数对象
        """

        if activation is not None:
            self._activation = activation

        if last_hop_activation is not None:
            self._last_hop_activation = last_hop_activation

        if loss is not None:
            self._loss = loss

    ''''''

    def train(self, sx_group_list, sy_group_list, loop_max, neuron_count_list, rate,
              init_from_unserialization=False, alpha_para=1, w_shape_list=None):
        """
        功能：神经网络训练\n
        参数：\n
        sx_group_list：分组训练样本输入列表\n
        sy_group_list：分组训练样本输出列表\n
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
        self._sx_group_list = sx_group_list
        self._sy_group_list = sy_group_list
        self._loop_max = loop_max
        self._rate = rate
        self._init_from_unserialization = init_from_unserialization
        self._alpha_para = alpha_para

        # 如果是卷积网络，这个参数没有意义（如果是卷积网络，直接传入 None 即可）
        self._neuron_count_list = neuron_count_list

        # 如果不是卷积网络，这个参数，没有意义（如果不是卷积网络，直接传入默认值即可）
        self._w_shape_layer = w_shape_list

        # 2. 校验
        err = self._valid()
        if errorcode.SUCCESS != err:
            print("\nvalid error, errcode = %d\n" % err)
            return err

        # 3. 初始化 w, b，及其他参数
        self._init_para()

        # 4. 训练
        return self._train()

    ''''''

    def _valid(self):
        """
        参数校验
        :return: error code
        """

        # 1. 校验每层神经元
        err = self._valid_layer_neuron()
        if errorcode.SUCCESS != err:
            print("_valid_layer_neuron error, err = %d" % err)
            return err

        # 2. 输入样本与输出样本
        err = self._valid_sample()
        if errorcode.SUCCESS != err:
            print("_valid_sample error, err = %d" % err)
            return err

        # 3. 最大循环训练次数，须 >= 1
        if 1 > self._loop_max:
            print("loop max error, loop_max = %d" % self._loop_max)
            return errorcode.FAILED

        return errorcode.SUCCESS

    ''''''

    def _valid_layer_neuron(self):
        """
        校验每层神经元
        :return: error code
        """

        # 1. 神经网络层数，须 >= 1
        layer_count = len(self._neuron_count_list)

        if 1 > layer_count:
            return errorcode.FAILED

        # 2. 每层的神经元个数，须 >= 1
        for layer in range(0, layer_count):
            count = self._neuron_count_list[layer]

            if 1 > count:
                return errorcode.FAILED

        return errorcode.SUCCESS

    ''''''

    def _valid_sample(self):
        """
        校验训练样本
        :return: error code
        """

        # 1. 训练样本的输入和输出，分组数量须相同
        len1 = len(self._sx_group_list)
        len2 = len(self._sy_group_list)

        if len1 != len2:
            return errorcode.FAILED

        # 2. 校验每一组训练样本
        for i in range(0, len1):
            err = self._valid_sample_sub(self._sx_group_list[i], self._sy_group_list[i])

            if errorcode.SUCCESS != err:
                return err

        return errorcode.SUCCESS

    ''''''

    def _valid_sample_sub(self, sx_list, sy_list):
        """
        校验样本
        :param sx_list: 输入样本列表
        :param sy_list: 输出样本列表
        :return: error code
        """

        # 1. 输入样本的数量与输出样本的数量，须相同
        len1 = len(sx_list)
        len2 = len(sy_list)

        if len1 != len2:
            return errorcode.FAILED

        # 2. 样本数量，须 >= 1
        sample_count = len1
        if 1 > sample_count:
            return errorcode.FAILED

        # 3. 样本向量维度

        # 输入向量维度
        sx_dim = len(sx_list[0])

        # 输出向量维度
        layer_count = len(self._neuron_count_list)
        sy_dim = self._neuron_count_list[layer_count - 1]

        # 3.1 输入样本/输出样本，向量维度 > 1
        if (1 > sx_dim) or (1 > sy_dim):
            return errorcode.FAILED

        # 3.2 每一个输入/输出样本的向量维度
        for i in range(0, sample_count):
            shape_in = sx_list[i].shape
            shape_out = sy_list[i].shape

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

    ''''''

    def _init_para(self):
        """
        初始化参数
        :return: error code
        """

        # 神经网络输入，向量维度
        sx_list = self._sx_group_list[0]
        self._sx_dim = len(sx_list[0])

        # 神经网络的层数
        self._layer_count = len(self._neuron_count_list)

        # 神经网络输出，向量维度
        self._sy_dim = self._neuron_count_list[self._layer_count - 1]

        # 初始化训练参数
        return self._init_train_para()

    ''''''

    def _init_train_para(self):
        """
        初始化训练参数
        :return: NULL
        """

        # 初始化 w，b
        self._init_w_b()

    ''''''

    def _init_w_b(self):
        """
        初始化 w，b 参数
        :return: error code
        """

        # 通过反序列化，初始化 w，b
        if self._init_from_unserialization:
            file_path = os.path.dirname(__file__) + "/../gl/train_para/"
            self._w_layer, self._b_layer = unserialize_train_para(file_path, self._layer_count, u_flag=False)
        # 通过随机值，初始化 w, b
        else:
            # 每一层 w、b 参数列表
            self._w_layer = list()
            self._b_layer = list()

            # 第1层 w 参数，w 是一个2维数组
            w = self._alpha_para * np.random.random((self._neuron_count_list[0], self._sx_dim))
            self._w_layer.append(w)

            # 第2层~第layer-1层 w 参数，w 是一个2维数组
            for i in range(1, self._layer_count):
                w = self._alpha_para * np.random.random((self._neuron_count_list[i], self._neuron_count_list[i - 1]))
                self._w_layer.append(w)

            # 第1层 ~ 第layer-1层 b 参数，b 是一个向量
            for i in range(0, self._layer_count):
                b = np.zeros([self._neuron_count_list[i], 1])
                self._b_layer.append(b)

        return errorcode.SUCCESS

    ''''''

    def _train(self):
        """
        训练
        :return: NULL
        """

        # 循环训练次数
        loop = 0

        # 打印开始时间
        localtime = get_local_time()
        print("\n\nbegin time = " + localtime + "\n")

        while 1:
            if loop >= self._loop_max:
                # 打印结束时间
                localtime = get_local_time()
                print("end time = " + localtime + "\n")

                # 打印最后一轮参数
                # self._print_train_para(loop)

                self._write_train_para(loop)

                break

            loop = loop + 1

            # 分组训练
            group_count = len(self._sx_group_list)

            # 分组训练，训练每一个样本
            for g in range(0, group_count):
                # 1. 每一组训练之前，预备工作
                self._pre_train()

                # 2. 获取分组训练样本
                sx_list = self._sx_group_list[g]
                sy_list = self._sy_group_list[g]

                sample_count = len(sx_list)

                # 3. 初始化训练参数的 delta
                self._init_train_para_delta()

                # 4. 针对该组的每一个样本开始训练
                for i in range(0, sample_count):
                    # 第 i 个训练样本
                    sx = sx_list[i]
                    sy = sy_list[i]

                    # 4.1 第 i 个训练样本，经过（多层）神经网络的计算
                    nn_y_list = self._calc_nn(sx)

                    # 4.2 最后一跳激活
                    nn_y = nn_y_list[len(nn_y_list) - 1]
                    last_hop_y = self._last_hop_activation.active_array(nn_y)
                    nn_y_list.append(last_hop_y)

                    # 4.3 根据神经网络计算结果，计算训练参数的 delta（比如：delta w, delta b）
                    self._calc_train_para_delta(nn_y_list, sx, sy)

                # 4.4 一组样本计算完毕，修正训练参数(比如：w, b)
                self._modify_train_para()

        return errorcode.SUCCESS

    ''''''

    def _pre_train(self):
        """
        每一轮训练之前预准备工作（一般来说，啥都不用做）
        :return: NULL
        """
        pass

    ''''''

    def _calc_nn(self, sx):
        """
        计算整个网络的输出
        :param sx: 神经网络的输入
        :return: 整个神经网络，每一层的输出
        """

        x = sx

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
        w = self._w_layer[layer]
        b = self._b_layer[layer]

        y = np.matmul(w, x) + b

        y = y + self._calc_recurrent(layer)

        # 激活
        y = self._activation.active_array(y)

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

    def _init_train_para_delta(self):
        """
        初始化训练参数的 delta
        :return: NULL
        """

        self._delta_w_layer = list()
        self._delta_b_layer = list()

        for i in range(0, self._layer_count):
            # _delta_w, _w 维度相同，初始值为 0
            _delta_w = np.zeros(self._w_layer[i].shape)
            self._delta_w_layer.append(_delta_w)

            # _delta_b, _b 维度相同，初始值为 0
            _delta_b = np.zeros(self._b_layer[i].shape)
            self._delta_b_layer.append(_delta_b)

    ''''''

    def _calc_train_para_delta(self, nn_y_list, sx, sy):
        """
        计算神经网络训练参数的 delta
        :param nn_y_list: 神经网络每一层的输出
        :param sx: 训练样本（输入）
        :param sy: 训练样本（输出）
        :return: NULL
        """
        pass

    ''''''

    def _modify_train_para(self):
        """
        根据练参数的 delta，修正训练参数
        :return: NULL
        """

        pass

    ''''''

    def _pre_predict(self):
        """
        预测前的准备工作
        :return: NULL
        """

        pass

    ''''''

    def predict(self, sx_list, revise_strong=False):
        """
        神经网络预测
        :param sx_list: 待预测的样本列表列表
        :param revise_strong: 预测时，修正标记
        :return: 预测结果
        """

        # 预测前，先做个准备工作
        self._pre_predict()

        # 开始预测

        count = len(sx_list)
        py_list = list()

        for i in range(0, count):
            sx = sx_list[i]
            nn_y_list = self._calc_nn(sx)

            # 最后一层的 nn_y，才是神经网络的最终输出
            nn_y = nn_y_list[len(nn_y_list) - 1]

            # 最后一跳激活
            lha_y = self._last_hop_activation.active_array(nn_y)

            # 最后一跳修正
            lhr_y = self._last_hop_activation.predict_revise(lha_y, revise_strong)

            # 然后再添加到预测列表
            py_list.append(lhr_y)

        return py_list

    ''''''

    def predict_recurrent(self, sx, py_list, max_recursion_count=30):
        """
        循环（递归）预测
        :param sx: 待预测样本
        :param py_list: 预测结果
        :param max_recursion_count: 最大递归次数
        :return: NULL
        """

        # 由于是递归调用，所以设置一个保护，防止死循环
        count = len(py_list)

        if count >= 30:
            return

        # 因为是递归调用，所以预测前的准备工作，只放在第一次预测时
        if 0 == count:
            self._pre_predict()

        nn_y_list = self._calc_nn(sx)

        # 最后一层的 nn_y，才是神经网络的最终输出
        nn_y = nn_y_list[len(nn_y_list) - 1]

        # 最后一跳激活
        lha_y = self._last_hop_activation.active_array(nn_y)

        # 最后一跳修正
        lhr_y = self._last_hop_activation.predict_revise(lha_y, revise_strong=True)

        # 对修正后的结果，再处理一次
        r_flag, ch, r_sx = self._handle_lhr(lhr_y)

        # 将 recurrent_sx 加入 py_list
        py_list.append(ch)

        # 如果需要递归，则继续递归预测
        if r_flag:
            self.predict_recurrent(r_sx, py_list, max_recursion_count)
        # 如果不需要递归，则啥都不做
        else:
            pass

    ''''''

    def _handle_lhr(self, lhr_y):
        """
        处理最后一跳修正后的输出
        :param lhr_y: 最后一跳修正后的输出
        :return: recurrent_flag，是否继续递归；recurrent_sx，如果递归，其 sx =  recurrent_sx
        """

        r_flag = False
        ch = "None"
        r_sx = None

        return r_flag, ch, r_sx

    ''''''

    def _print_train_para(self, loop):
        """
        打印 w, b, loop
        :param loop: 神经网络的训练轮次
        :return: NULL
        """

        # 新启一行
        print("\n")

        # 训练轮次
        print("训练轮次 = %d\n" % loop)

        # 训练参数
        train_para_str = self._create_train_para_string()

        print(train_para_str)

    ''''''

    def _write_train_para(self, loop):
        """
        将 w, b，loop 写入文件
        :param loop: 神经网络的训练轮次
        :return: NULL
        """

        # 1. 将训练参数以字符串形式保存在文件中
        self._write_train_para_string(loop)

        # 2. 将训练参数序列化到文件
        self._serialize_train_para()

    ''''''

    def _write_train_para_string(self, loop):
        """
        将训练参数以字符串形式保存在文件中
        :param loop: 神经网络的训练轮次
        :return: NULL
        """

        # 训练参数字符串
        train_para_str = ""

        # 记录时间
        localtime = get_local_time()
        train_para_str += "time = " + localtime + "\n\n"

        # 训练轮次
        train_para_str += "训练轮次 = %d\n\n" % loop

        # 训练参数
        train_para_str += self._create_train_para_string()

        # 写入文件
        file_name = os.path.dirname(__file__) + "/../" + self._para_file_path + "train_para.txt"
        with open(file_name, 'w', newline="\n", encoding='utf-8') as f:
            f.write(train_para_str)

    ''''''

    def _serialize_train_para(self):
        """
        将训练参数序列化到文件
        :return: NULL
        """

        file_path = os.path.dirname(__file__) + "/../" + self._para_file_path

        for layer in range(0, self._layer_count):
            # w 参数文件名
            w_file_name = file_path + "w%d" % layer
            # 序列化 w
            pickle.dump(self._w_layer[layer], open(w_file_name, 'wb'))

            # b 参数文件名
            b_file_name = file_path + "b%d" % layer
            # 序列化 b
            pickle.dump(self._b_layer[layer], open(b_file_name, 'wb'))

    ''''''

    def _create_train_para_string(self):
        """
        将训练参数转化为 string
        :return: 练参数转化后的 string
        """

        # 训练参数字符串
        train_para_str = ""

        # 构建每一层训练参数的字符串
        for layer in range(0, self._layer_count):
            # loop
            train_para_str += "layer ＝ %d\n\n" % layer

            # w
            train_para_str += "w%d:\n\n" % layer
            train_para_str += array_2_string(self._w_layer[layer])

            # b
            train_para_str += "\n\n"
            train_para_str += "b%d:\n\n" % layer
            train_para_str += array_2_string(self._b_layer[layer])

            # 换行
            train_para_str += "\n\n"

        return train_para_str

    ''''''

    def stub_set_para(self, sx_dim, neuron_count_list, w_layer, b_layer, activation):
        """

        :param sx_dim: 神经网络输入，向量维度
        :param neuron_count_list: 神经网络层数
        :param w_layer: 每一层 w 参数 列表，w 是个 matrix
        :param b_layer: 每一层 b 参数 列表，b 是个 vector
        :param activation: 激活函数
        :return: NULL
        """
        # 神经网络输入，向量维度
        self._sx_dim = sx_dim

        # 每一层神经元的数量(Neuron Count)
        self._neuron_count_list = neuron_count_list

        # 神经网络层数
        self._layer_count = len(w_layer)

        # 每一层 w 参数，w 是个 matrix
        self._w_layer = w_layer

        # 每一层 b 参数，b 是个 vector
        self._b_layer = b_layer

        # 激活函数对象
        self._activation = activation
