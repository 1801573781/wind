"""
Function：测试写诗机
Author：lzb
Date：2021.03.05
"""

import numpy as np

from activation.last_hop_activation import SoftMaxLHA
from activation.normal_activation import Sigmoid, ReLU, LeakyReLU

from loss.loss import CrossEntropyLoss
from rnn import rnn_poem_recitation
from rnn.rnn_poet import Poet
from sample.one_poem_sample import OnePoemSample
from sample.poem_sample import PoemSample


def test_poet():
    """
    测试 rnn_poet（写诗机的第一步）
    :return: NULL
    """

    # 1. 构建神经网络对象

    # 激活函数
    # activation = Sigmoid()
    # activation = ReLU(20)
    activation = LeakyReLU(20)

    # 最后一跳激活函数
    last_hop_activation = SoftMaxLHA()

    # 损失函数
    loss = CrossEntropyLoss()

    # 神经网络
    nn = Poet(activation, last_hop_activation, loss)

    # 2. 构建训练样本

    # 训练样本对象
    sample = PoemSample()

    sample.create_sample_group()

    train_sx_group = sample.get_sx_group()
    train_sy_group = sample.get_sy_group()

    # 3. 训练

    # 每一层网络的神经元个数
    neuron_count_list = [10, 21]

    # 最大循环训练次数
    loop_max = 20000

    # 学习效率
    rate = 0.1

    # 训练
    nn.train(train_sx_group, train_sy_group, loop_max, neuron_count_list, rate)

    # 4. 测试

    # 创建测试样本
    # 诗的起始字符
    ch = "白"
    # ch = "床"
    test_sx = sample.create_test_sample(ch)

    # 测试
    py_list = list()
    nn.predict_recurrent(test_sx, py_list)

    # 将测试样本放在首位，这样就组成了一首完整的诗
    py_list.insert(0, ch)

    print("\n")
    print("py:\n")

    print(py_list)


''''''


def test_poet_without_train():
    """
    测试 rnn，不经过训练，直接赋值参数
    :return: NULL
    """

    # 字符，决定选哪一首诗
    # ch = "白"
    ch = "床"

    # 激活函数
    activation = LeakyReLU(20)

    # 最后一跳激活函数
    last_hop_activation = SoftMaxLHA()

    # 损失函数
    loss = CrossEntropyLoss()

    # 神经网络
    nn = rnn_poem_recitation.PoemRecitation(activation, last_hop_activation, loss, ch)

    # 每一层网络的神经元个数
    neuron_count_list = [10, 21]

    # 每一层 w、B 参数，w 是个 matrix，b 是个 vector（数据类型也是一个 matrix）
    w_layer = list()
    b_layer = list()
    u_layer = list()

    # 第0层

    w0 = np.asarray([
        [-3.538559, 9.859105, 9.893933, -11.991461, 4.702758, -15.647867, -10.492320, -10.433017,
         -5.056030, 9.242447, -1.314108, -4.134307, 14.202259, 0.427581, 0.744011, -13.722976, 0.424487,
         13.678333, -5.848754, -2.227606, 0.117229],

        [-0.660331, 9.028281, -5.892644, 13.522159, -8.231236, -2.606620, -9.846532, -19.122620, -2.896445,
         2.964697, 13.335625, 10.311550, 13.514065, 0.874618, 0.451721, -0.836880, 0.496568, -8.397640,
         -1.518009, -11.913896, 0.049554],

        [-6.792985, -0.352270, -19.883430, 29.817777, -11.435967, -3.547367, -8.224099, 3.401731,
         -4.560222, 22.245467, -2.432454, 3.531877, -1.051416, 0.661556, 0.430565, -13.373473, 0.859542,
         7.337146, -13.711478, -1.358087, 0.317916],

        [-7.718142, 6.194956, -4.591991, 3.366489, -13.346880, -7.899646, 10.286536, -1.271208, 0.473965,
         8.426327, -5.395472, -16.283728, 8.175877, 0.183308, 0.426683, -3.749780, 0.875708, -3.891130,
         3.436521, -4.681474, 0.793848],

        [5.945570, -7.855750, 0.691770, -0.008717, 8.736372, 5.225561, -0.486366, 8.297081, -10.550413,
         -3.183117, 1.417966, 11.858798, -7.916924, 0.908711, 0.519771, -0.149767, 0.236339, -6.512125,
         -0.899299, -7.507509, 0.623309],

        [2.690629, -3.334328, -6.554794, -1.870296, -1.121759, -3.793306, -6.259013, 3.318526, 12.390852,
         -5.998210, 13.011822, -3.974253, -3.128111, 0.125356, 0.592506, 10.329325, 0.834903, -8.795272,
         -7.135564, -1.600805, 0.338428],

        [-3.957669, 1.357045, -15.043584, 12.395384, 8.803891, -7.905241, -2.009854, -4.035662, -15.659553,
         7.610266, 10.650116, -20.873474, 1.447397, 0.621858, 0.992691, -0.823846, 0.827302, -1.567133,
         5.755539, 5.889399, 0.294997],

        [-6.344592, -3.097812, -9.443741, -7.340244, 0.569867, 0.416727, 7.340264, 5.678316, -7.456655,
         15.790775, 8.232506, -2.392731, -3.529393, 0.749220, 0.076259, -3.898968, 0.957944, -8.495143,
         -15.854053, 8.262181, 0.405422],

        [-6.528152, -6.672878, -10.092714, -4.121259, -5.973298, 16.523494, 4.663862, -9.762958, -0.443804,
         20.706089, -1.408898, -5.886947, -8.049727, 0.952621, 0.357359, 6.011714, 0.174756, -6.773154,
         -3.783480, 2.302081, 0.259373],

        [-0.079871, 1.541042, 6.469698, 6.931514, -5.306168, 1.792247, -8.443862, 2.882729, 2.902644,
         0.916217, -3.223143, -19.890231, 4.381905, 0.884483, 0.699021, -14.496110, 0.581673, -11.609024,
         5.500725, 13.208895, 0.544432]
    ])

    b0 = np.asarray([[-31.668994],
                     [-16.752892],
                     [-27.953559],
                     [-38.347040],
                     [-11.464668],
                     [-20.412068],
                     [-26.005257],
                     [-30.975978],
                     [-26.400265],
                     [-25.564401]])

    u0 = np.asarray(
        [[0.591729, 0.955378, 0.906886, 0.895575, 0.022914, 0.877011, 0.716369, 0.447997, 0.254436, 0.532483],
         [0.926068, 0.869491, 0.940422, 0.268933, 0.549318, 0.058505, 0.473500, 0.688068, 0.352167, 0.822833],
         [0.499318, 0.465786, 0.271015, 0.813262, 0.015166, 0.787538, 0.123711, 0.479582, 0.189888, 0.285797],
         [0.521010, 0.355995, 0.872293, 0.547698, 0.358661, 0.135227, 0.749628, 0.672987, 0.974619, 0.239665],
         [0.582553, 0.873779, 0.817656, 0.114562, 0.115217, 0.271238, 0.849649, 0.845323, 0.080006, 0.555832],
         [0.460684, 0.119094, 0.649794, 0.946176, 0.230993, 0.260399, 0.724224, 0.423979, 0.652385, 0.822949],
         [0.616026, 0.635836, 0.794412, 0.864816, 0.068226, 0.075906, 0.012071, 0.205577, 0.757503, 0.385775],
         [0.148718, 0.042357, 0.505948, 0.794862, 0.547210, 0.256621, 0.311074, 0.098121, 0.747385, 0.747731],
         [0.678199, 0.790660, 0.077679, 0.350564, 0.213652, 0.441854, 0.426849, 0.334336, 0.007756, 0.100748],
         [0.964516, 0.088888, 0.127946, 0.993677, 0.547584, 0.849531, 0.746167, 0.596135, 0.009402, 0.822361]])

    w_layer.append(w0)
    b_layer.append(b0)
    u_layer.append(u0)

    # 第1层

    w1 = np.asarray(
        [[2.625108, 3.076668, 2.479864, 2.921179, 1.213516, 2.120475, 1.693972, 1.509206, 1.677587, 2.511474],
         [3.780800, 6.360721, -7.136901, -4.054355, 13.814681, 9.441299, -0.182720, -4.095318, -4.874726, 7.086200],
         [15.921735, 15.535968, -5.904895, 7.915341, -11.445816, -5.482622, -0.682130, -5.873845, -10.803492, 3.583040],
         [16.915842, 0.822590, -13.601553, 2.842296, 4.110998, -1.626120, -7.878896, -4.960412, -4.931569, 13.283947],
         [-10.070944, 12.411449, 23.696768, -0.562570, 0.027888, -2.501077, 10.147573, -8.365579, -3.278798, 8.734748],
         [8.870132, -4.798870, -10.479113, -7.480070, 10.063162, 3.583815, 13.903520, 3.473117, -3.341442, -1.445798],
         [-10.404631, -1.120865, -5.304975, -1.756465, 5.899682, 0.631952, -3.684715, 4.048065, 19.875093, 8.674242],
         [-5.717255, -6.623915, -8.084196, 14.770370, 0.445464, -2.566361, 1.584923, 11.559912, 6.802996, -5.559356],
         [-4.698887, -16.933952, 3.167116, 3.485993, 9.576927, 7.462189, -1.367081, 10.261619, -6.701490, 6.620207],
         [0.517802, -0.833780, -4.185347, 5.701596, -9.539171, 16.460860, -11.558244, -2.731062, 2.224711, 8.490337],
         [3.741092, -6.623729, 10.903142, 0.792890, -9.399227, -10.801432, 0.049167, 11.542444, 15.602948, -7.481282],
         [-7.592851, 9.321826, -11.989424, -0.871858, -0.060757, 18.768256, 11.133969, 3.795271, 4.002577, -10.036868],
         [7.789190, 18.354111, 3.265507, -9.591738, 22.673436, 5.118205, -12.712800, -0.377375, 2.644977, -8.605718],
         [3.123294, 2.089489, 2.552923, 3.137642, 1.077771, 1.644145, 2.208192, 1.601827, 1.269330, 2.887080],
         [2.980628, 2.879987, 1.692504, 2.978769, 1.419491, 2.128661, 1.651063, 1.780276, 1.838924, 2.468141],
         [-3.500142, -7.994071, 9.010634, 8.551510, -10.665313, -4.284765, 5.878478, 3.163462, 2.518347, -1.651620],
         [2.729233, 2.925245, 2.475429, 2.172573, 1.404497, 2.246705, 2.457081, 1.653783, 1.372623, 2.547688],
         [3.281696, 0.923317, 1.535836, -0.178427, 1.472406, 1.512333, -5.667282, 9.267400, 2.148267, -9.853637],
         [21.413085, -6.983151, 3.235056, 3.721981, -6.240900, -5.540742, 3.953269, -4.499009, -2.446553, -6.084940],
         [2.195003, 2.452120, -15.702378, 8.239252, 2.109656, -2.483901, 8.631467, -10.312351, 0.127620, 9.137576],
         [0.902036, -12.302542, -4.991936, -3.828270, -7.519668, 0.215434, 5.425067, 10.650406, 3.612202, 13.587027]])

    b1 = np.asarray([[-0.676784],
                     [6.016811],
                     [3.786305],
                     [-2.741812],
                     [7.076487],
                     [0.353375],
                     [-0.937339],
                     [-1.228204],
                     [-1.666120],
                     [-2.280538],
                     [2.530664],
                     [0.766568],
                     [1.060229],
                     [-0.667260],
                     [-0.668722],
                     [-4.517910],
                     [-0.665446],
                     [-4.699517],
                     [-3.230313],
                     [-3.102659],
                     [-1.157181]])

    # u1 暂时没用到，赋值为0即可
    u1 = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    w_layer.append(w1)
    b_layer.append(b1)
    u_layer.append(u1)

    # nn 设置参数
    nn.stub_set_para(neuron_count_list, w_layer, b_layer, u_layer)

    # 创建测试样本
    sample = OnePoemSample(ch)
    test_sx = sample.create_test_sample(ch)

    # 测试
    py_list = list()
    nn.predict_recurrent(test_sx, py_list)

    # 将测试样本放在首位，这样就组成了一首完整的诗
    py_list.insert(0, ch)

    print("\n")
    print("py:\n")

    print(py_list)
