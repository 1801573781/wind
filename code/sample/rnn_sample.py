"""
Function：rnn 训练样本
Author：lzb
Date：2021.02.19

说明：
1、先简单构建1首诗的训练样本：
床前明月光
疑是地上霜
举头望明月
低头思故乡
2、测试样本，也用这首诗
"""
from gl.hanzi_encoder import HanziEncoder
from sample.fully_connected_sample import FullConnectedSample
from gl.matrix_list import list_2_matrix


class RNNSanmple(FullConnectedSample):
    """
    简单测试，只用李白这首诗做 one-hot 编码，构建 rnn 训练样本
    """

    poem = ["床", "前", "明", "月", "光",
            "疑", "是", "地", "上", "霜",
            "举", "头", "望", "明", "月",
            "低", "头", "思", "故", "乡",
            "END"
            ]

    poem_encoder = HanziEncoder()

    ''''''

    def create_sample(self):
        """
        创建样本
        :return: NULL
        """

        self._create_sx_list()
        self._create_sy_list()

    ''''''

    def _create_sx_list(self):
        """
        创建样本-输入
        :return: NULL
        """
        self._sx_list = list()

        count = len(self.poem)

        for i in range(0, count - 1):
            sx = self.poem_encoder.encode(self.poem[i])
            sx = list_2_matrix(sx)
            self._sx_list.append(sx)

    ''''''

    def _create_sy_list(self):
        """
        创建样本-输出
        :return: NULL
        """
        self._sy_list = list()

        count = len(self.poem)

        for i in range(1, count):
            sy = self.poem_encoder.encode(self.poem[i])
            sy = list_2_matrix(sy)
            self._sy_list.append(sy)

    ''''''

    def create_test_sample(self, ch):
        """
        创建测试样本（输入）
        :param ch: 测试样本字符
        :return: ch 的 one-hot 编码
        """
        sx = self.poem_encoder.encode(ch)
        return sx


def test():
    sample = RNNSanmple()
    sample.create_sample()

    sx_list = sample.get_sx_list()
    sy_list = sample.get_sy_list()

    print(sx_list)
    print(sy_list)
