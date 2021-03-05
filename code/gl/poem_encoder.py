"""
Function：古诗汉字 one-hot 编码
Author：lzb
Date：2021.03.05

说明：选取了部分古诗作为汉字集
"""

from gl.hanzi_encoder import HanziEncoder


class PoemEncoder(HanziEncoder):
    """
    选取了部分古诗作为汉字集
    """

    # 相当于 C++ 的静态变量
    _instance = None

    # 静态函数，创建 PoemEncoder 的实例（单例模式，不考虑多线程。不过，Python 也没有多线程，^_^）
    @staticmethod
    def instance(poem_file=None):
        if PoemEncoder._instance is None:
            PoemEncoder._instance = PoemEncoder(poem_file)

        return PoemEncoder._instance

    ''''''

    # 古诗所在文件
    _poem_file = "./poem/poem.txt"

    ''''''

    def __init__(self, poem_file=None):
        """
        构造函数
        :param poem_file: 诗歌文件
        """

        if poem_file is not None:
            self._poem_file = poem_file

        super().__init__()

    ''''''

    def _init_dict(self):
        """
        初始化汉字编码字典。
        :return: NULL
        """

        # 读取古诗文件(并做预处理)
        poem_data = self._read_poem()

        # 构建汉字编码字典
        self._init_dict_by_poem(poem_data)

    ''''''

    def _read_poem(self):
        """
        读取古诗，并作预处理
        :return: 预处理后的数据
        """

        # 读取古诗文件
        with open(self._poem_file, "r", encoding='utf-8') as f:  # 打开文件
            data = f.read()  # 读取文件

        # 去除重复字符。将 string 转成 set 就可以去重
        data = set(data)

        # 再将 set 转为 string
        data = "".join(data)

        # 将 “#” 替换为 “”
        data = data.replace("#", "")

        # 将 "\n" 替换为 “”
        data = data.replace("\n", "")

        # 最后加一个 END 字符
        data = data + HanziEncoder.END

        return data

    ''''''

    def _init_dict_by_poem(self, poem_data):
        """
        根据处理后的诗歌汉字，构建汉字编码字典
        :param poem_data: 处理后的诗歌汉字
        :return: NULL
        """

        # 汉字编码字典初始化
        self._dict = dict()

        # 诗歌汉字数量
        count = len(poem_data)

        # 开始编码
        for i in range(0, count):
            # one-hot 编码
            one_hot = [0] * count
            one_hot[i] = 1

            # 汉字字符
            ch = poem_data[i]

            # 插入字典
            self._dict[ch] = one_hot


''''''


def test():
    poem_encoder = PoemEncoder.instance()

    print(poem_encoder._dict)

    return poem_encoder
