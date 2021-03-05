"""
Function：古诗训练样本
Author：lzb
Date：2021.03.05
"""

from gl.hanzi_encoder import HanziEncoder
from gl.hanzi_encoder_simple import HanziEncoderSimple
from gl.poem_encoder import PoemEncoder
from sample.fully_connected_sample import FullConnectedSample
from gl.matrix_list import list_2_matrix


class PoemSample(FullConnectedSample):
    """
    选取了部分古诗作为训练样本
    """

    # 古诗所在文件
    _poem_file = "./../gl/poem/poem.txt"

    # 古诗分隔符
    _split = "#"

    # 输入样本组
    _sx_group = None

    # 输出样本组
    _sy_group = None

    ''''''

    def create_sample_group(self):
        """
        创建样本组
        :return: NULL
        """

        # 如果 _sx_group 不为空，说明已经创建过了，就不必再创建
        if self._sx_group is not None:
            return
        # 否则的话，就开始创建
        else:
            # 初始化 self._sx_group, self._sy_group
            self._sx_group = list()
            self._sy_group = list()

            # 读取古诗
            poems = self._read_poems()

            # 针对每一首古诗，构建样本
            count = len(poems)

            for i in range(0, count):
                self._create_one_poem_sample(poems[i])

    ''''''

    def _read_poems(self):
        """
        创建样本组，内部函数
        :return: NULL
        """

        # 读取古诗文件
        with open(self._poem_file, "r", encoding='utf-8') as f:  # 打开文件
            data = f.read()  # 读取文件

        # 以 “#” 分割 data
        poems = data.split(self._split, -1)

        return poems

    ''''''

    def _create_one_poem_sample(self, poem):
        """
        创建一首诗的训练样本
        :param poem: 一首诗
        :return: NULL
        """

        # 1. 先对这首诗做个预处理

        # 1.1 将 "\n" 替换为 “”
        poem = poem.replace("\n", "")

        # 1.2 最后加一个 END 字符
        poem = poem + HanziEncoder.END

        # 2. 构建样本

        # 2.1 初始化样本列表 sx_list, sy_list
        sx_list = list()
        sy_list = list()

        # 2.2 构建样本列表
        count = len(poem)

        for i in range(0, count - 1):
            sx = PoemEncoder.instance().encode(poem[i])
            sx = list_2_matrix(sx)
            sx_list.append(sx)

            sy = PoemEncoder.instance().encode(poem[i + 1])
            sy = list_2_matrix(sy)
            sy_list.append(sy)

        # 2.3 将样本列表加入样本分组
        self._sx_group.append(sx_list)
        self._sy_group.append(sy_list)

    ''''''

    def get_sx_group(self):
        """
        获取输入样本组
        :return: self._sx_group
        """

        return self._sx_group

    ''''''

    def get_sy_group(self):
        """
        获取输出样本组
        :return: self._sy_group
        """

        return self._sy_group


''''''


def test():
    """
    测试 PoemSample
    :return: NULL
    """

    ps = PoemSample()

    ps.create_sample_group()

    sx_group = ps.get_sx_group()
    sy_group = ps.get_sy_group()

    print(sx_group)
    print(sy_group)
