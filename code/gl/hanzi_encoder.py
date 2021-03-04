"""
Function：汉字 one-hot 编码
Author：lzb
Date：2021.03.05
"""


class HanziEncoder:
    """
    汉字 one-hot 编码, base class
    """

    # end 字符
    _end = "END"

    _dict = None

    ''''''

    def __init__(self):
        """
        构造函数
        """

        # 初始化汉字编码字典
        self._init_dict()

    ''''''

    def _init_dict(self):
        """
        初始化汉字编码字典。虚函数，待子类继承
        :return: NULL
        """

        pass

    def encode(self, ch):
        """
        将一个汉字 or "END" 编码为 one-hot
        :param ch: 汉字 or "END"
        :return: ch 对应的 one-hot 编码
        """

        if ch in self._dict.keys():
            return self._dict[ch]
        else:
            # 需要记录日志
            print("\nthere has no ch in dict, ch = %s\n" % ch)

            return self._dict[self._end]

    ''''''

    def decode(self, ec):
        """
        将 one-hot 编码，解码为一个汉字 or ”END“
        :param ec: one hot 编码(一个 list)
        :return: ec 对应的汉字（或者 ”END“）
        """

        idx = list(self._dict.values()).index(ec)

        if idx >= 0:
            return list(self._dict.keys())[idx]
        else:
            # 需要记录日志
            print("\nthere has no ec in dict, ec =\n")
            print(ec)

            return self._end

    ''''''

    def is_end(self, ch):
        """
        判断一个字符，是否是 "END"
        :param ch: 字符
        :return: true or false
        """

        if ch == self._end:
            return True
        else:
            return False
