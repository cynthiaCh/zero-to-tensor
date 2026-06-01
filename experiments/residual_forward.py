"""用 NumPy 演示最小残差连接：y = F(x) + x。"""

import numpy as np


def relu(x):
    """ReLU 激活函数：小于 0 的值变成 0，其余保持不变。"""
    return np.maximum(0, x)


def linear(x, weight, bias):
    """最小全连接层：输入乘权重，再加偏置。"""
    return x @ weight + bias


def residual_block_forward(x, w1, b1, w2, b2):
    """两层 MLP 分支加一条捷径分支，演示残差块的前向传播。"""
    # 主分支 F(x)：先做一次线性变换，再过 ReLU，再做一次线性变换。
    hidden = relu(linear(x, w1, b1))
    branch = linear(hidden, w2, b2)

    # 捷径分支 shortcut：把输入 x 原样加回来。
    y = branch + x
    return y, branch


def main():
    # x 可以先理解成“一条样本的 4 个特征”，暂时不用想图片。
    x = np.array([[1.0, -2.0, 0.5, 3.0]])

    # 为了能和 x 相加，F(x) 的输出形状必须也是 (1, 4)。
    w1 = np.eye(4)
    b1 = np.array([0.0, 2.5, 0.0, -1.0])
    w2 = np.array(
        [
            [0.2, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.2],
        ]
    )
    b2 = np.zeros(4)

    y, branch = residual_block_forward(x, w1, b1, w2, b2)

    print("输入 x:")
    print(x)
    print("主分支 F(x):")
    print(branch)
    print("残差输出 y = F(x) + x:")
    print(y)
    print("x.shape:", x.shape)
    print("F(x).shape:", branch.shape)
    print("y.shape:", y.shape)


if __name__ == "__main__":
    main()
