"""
Function：感知器测试
Author：lzb
Date：2021.01.02
"""


from gl import draw
from sample import straight_line_sample
from nn.test_nn import NNTest
from pt.perceptron import Perceptron


"""
功能：测试 ax + b 分割
参数：NULL    
返回值：NULL
"""


def test_straight_line():
    # 训练样本个数
    train_sample_count = 1000

    # 训练样本，输入，每个向量元素的最大值
    train_sx_max = list()
    train_sx_max.append(200)
    train_sx_max.append(500)

    # ax + b 分割的样本对象
    sample = straight_line_sample.StraightLineSample()

    # 感知器对象
    nn = Perceptron()

    # 每一层网络的神经元个数（感知器只有1层，只有1个神经元）
    neuron_count_list = [1]

    # 最大循环训练次数
    loop_max = 10

    # 学习效率
    rate = 0.15

    # 预测样本参数
    predict_sample_count = 500

    # 预测样本，输入，每个向量元素的最大值
    predict_sx_max = list()
    predict_sx_max.append(2000)
    predict_sx_max.append(5000)

    # title
    title = "感知器"

    # 是否画训练样本
    draw_train_sample_flag = draw.ShowFlag.NO_SHOW

    # 是否画预测样本
    draw_predict_sample_flag = draw.ShowFlag.NO_SHOW

    # 是否画预测结果
    draw_predict_result_flag = draw.ShowFlag.SHOW

    test = NNTest()

    test.test(train_sample_count, train_sx_max, sample, neuron_count_list, loop_max, rate,
              predict_sample_count, predict_sx_max, nn, title,
              draw_train_sample_flag, draw_predict_sample_flag, draw_predict_result_flag)

