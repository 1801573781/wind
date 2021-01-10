"""
Function：神经网络测试
Author：lzb
Date：2021.01.07
"""

from gl import draw
from gl import common_function
from activation import active


"""
class：NNTest，神经网络测试
"""


class NNTest:
    # 训练样本，输入，向量维度
    sx_dim = 2

    # 训练样本，输出，向量维度
    sy_dim = 1

    # 激活函数对象
    activation = active.Sigmoid()

    """
    功能：测试神经网络
    参数：测试参数（略）      
    返回值：NULL
    """

    def test(self, train_sample_count, train_sx_max, sample, neuron_count_list, loop_max, rate,
             predict_sample_count, predict_sx_max, nn, title,
             draw_train_sample_flag=draw.ShowFlag.NO_SHOW,
             draw_predict_sample_flag=draw.ShowFlag.NO_SHOW,
             draw_predict_result_flag=draw.ShowFlag.SHOW,
             sx_dim=2, sy_dim=1):

        # 1. 成员变量赋值
        self.sx_dim = sx_dim
        self.sy_dim = sy_dim

        # 2. 创建训练样本
        sample.create_sample(train_sample_count, train_sx_max, self.sx_dim, self.sy_dim)

        if draw_train_sample_flag.value:
            sample.draw_sample(title + ": train sample")

        sx_list = sample.get_sx_list()
        sy_list = sample.get_sy_list()

        # 3. 训练
        nn.train(sx_list, sy_list, loop_max, neuron_count_list, rate, self.activation)

        # 4. 预测

        # 4.1 创建预测样本
        sample.create_sample(predict_sample_count, predict_sx_max, self.sx_dim, self.sy_dim)

        if draw_predict_sample_flag.value:
            sample.draw_sample(title + ": predict sample")

        sx_list = sample.get_sx_list()
        sy_list = sample.get_sy_list()

        # 4.2 预测
        py_list = nn.predict(sx_list, sy_list)

        accuracy = common_function.calculate_accuracy(py_list, sy_list)

        title = title + ": predict result, accuracy = %.3f%%" % (accuracy * 100)

        if draw_predict_result_flag.value:
            draw.draw_predict(title, sx_list, py_list, sample)

    """
    功能：测试神经网络，不经过训练，直接赋值参数
    参数：测试参数（略）      
    返回值：NULL
    """

    def test_stub(self, predict_sample_count, predict_sx_max, neuron_count_list, W, B, sample, nn, title,
                  draw_predict_sample_flag=draw.ShowFlag.NO_SHOW,
                  draw_predict_result_flag=draw.ShowFlag.SHOW,
                  sx_dim=2, sy_dim=1):
        # 1. 成员变量赋值
        self.sx_dim = sx_dim
        self.sy_dim = sy_dim

        # 2. 参数赋值
        nn.stub_set_para(sx_dim, neuron_count_list, W, B, self.activation)

        # 3. 预测

        # 3.1 创建预测样本
        sample.create_sample(predict_sample_count, predict_sx_max, self.sx_dim, self.sy_dim)

        if draw_predict_sample_flag.value:
            sample.draw_sample(title + ": predict sample")

        sx_list = sample.get_sx_list()
        sy_list = sample.get_sy_list()

        # 3.2 预测
        py_list = nn.predict(sx_list, sy_list)

        accuracy = common_function.calculate_accuracy(py_list, sy_list)

        title = title + ": predict result, accuracy = %.3f%%" % (accuracy * 100)

        if draw_predict_result_flag.value:
            draw.draw_predict(title, sx_list, py_list, sample)
