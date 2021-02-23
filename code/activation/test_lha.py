"""
Function：测试最后一跳激活函数
Author：lzb
Date：2021.01.31
"""

import numpy as np

from activation.last_hop_activation import LastHopActivation, DichotomyLHA, SoftMaxLHA


def test_last_hop_activation():
    lha = DichotomyLHA()

    nn_y = np.asarray([[[0.496053], [0.142468], [0.692607]],
                     [[-0.152569], [0.746855], [0.665359]],
                     [[-0.078141], [0.187796], [0.004870]]])

    lha.active(nn_y)

    print("\nDichotomyLHA: train\n")

    print(nn_y)

    lha.active(nn_y)

    print("\nDichotomyLHA: predict\n")

    print(nn_y)

    lha = SoftMaxLHA()

    lha.active(nn_y)

    print("\nSoftMaxLHA: train\n")

    print(nn_y)

    lha.active(nn_y)

    print("\nSoftMaxLHA: predict\n")

    print(nn_y)





