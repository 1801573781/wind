"""
Function：汉字 one-hot 编码
Author：lzb
Date：2021.02.20

说明：先简单构建1首诗的汉字编码：
床前明月光
疑是地上霜
举头望明月
低头思故乡
"""


class HanziEncoder:
    """
    简单测试，只用李白这首诗做 one-hot 编码
    """

    # end 字符
    _end = "END"

    # 这首诗中有几个重复的字符：明、月、头，将这几个重复字符换成：唐、李、白
    _dict = {
        "床": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "前": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "明": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "月": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "光": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "疑": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "是": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "地": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "上": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "霜": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "举": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "头": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "望": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "唐": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "李": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "低": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "白": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "思": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "故": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "乡": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        _end: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }

    ''''''

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
