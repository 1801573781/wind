"""
Function：通过循环神经网络，写诗(或者其他文字)
Author：lzb
Date：2021.03.03
"""


from gl.matrix_list import matrix_2_list, list_2_matrix
from gl.poem_encoder import PoemEncoder
from rnn.rnn_ex import RnnEx


class Poet(RnnEx):
    """
    通过循环神经网络，写诗(或者其他文字)
    """

    # 汉字编码解码器
    _hanzi_encoder = PoemEncoder.instance()

    ''''''

    def _handle_lhr(self, lhr_y):
        """
        处理最后一跳修正后的输出
        :param lhr_y: 最后一跳修正后的输出
        :return: recurrent_flag，是否继续递归；recurrent_sx，如果递归，其 sx =  recurrent_sx
        """

        # 将矩阵 lhr_y 转成 list
        lst = matrix_2_list(lhr_y)

        # 解码
        ch = self._hanzi_encoder.decode(lst)

        # 如果 ch == END，那么结束递归
        if self._hanzi_encoder.is_end(ch):
            r_flag = False
            r_sx = None
        # 否则，递归下去
        else:
            # 将 ch 编码
            r_sx = self._hanzi_encoder.encode(ch)
            # 将 r_sx 转换为矩阵
            r_sx = list_2_matrix(r_sx)

            r_flag = True

        return r_flag, ch, r_sx
