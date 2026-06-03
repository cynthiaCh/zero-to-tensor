"""两层线性网络的前向传播演示。

这一步先不训练，只观察数据怎么从 x 传到 hidden，再传到输出。
"""

import numpy as np


def make_demo_input():
    """构造 3 条一维样本：每一行是一条样本。"""
    return np.array([[1.0], [2.0], [3.0]])


def make_demo_params():
    """构造两层线性网络的固定参数，方便手算。"""
    return {
        "w1": np.array([[2.0]]),
        "b1": np.array([[0.0]]),
        "w2": np.array([[3.0]]),
        "b2": np.array([[0.0]]),
    }


def linear(x, weight, bias):
    """线性层：输入乘权重，再加偏置。"""
    return x @ weight + bias


def two_layer_branch(x, params):
    """两层主分支 F(x)：x -> hidden -> branch。"""
    # 第 1 层：把输入 x 变成中间结果 hidden。
    hidden = linear(x, params["w1"], params["b1"])

    # 第 2 层：把 hidden 继续变成主分支输出 branch。
    branch = linear(hidden, params["w2"], params["b2"])
    return branch, hidden


def plain_forward(x, params):
    """普通两层线性网络：输出只等于 F(x)。"""
    branch, hidden = two_layer_branch(x, params)
    y_pred = branch
    return y_pred, branch, hidden


def residual_forward(x, params):
    """残差两层线性网络：输出等于 x + F(x)。"""
    branch, hidden = two_layer_branch(x, params)
    y_pred = x + branch
    return y_pred, branch, hidden


def print_step(name, value):
    """打印值和 shape，方便观察数据传播。"""
    print(name)
    print(value)
    print("shape:", value.shape)
    print()


def main():
    x = make_demo_input()
    params = make_demo_params()

    plain_y, plain_branch, plain_hidden = plain_forward(x, params)
    residual_y, residual_branch, residual_hidden = residual_forward(x, params)

    print("两层线性网络前向传播")
    print("参数：w1=2, b1=0, w2=3, b2=0")
    print()
    print_step("输入 x", x)
    print_step("第 1 层 hidden = x @ w1 + b1", plain_hidden)
    print_step("第 2 层 F(x) = hidden @ w2 + b2", plain_branch)
    print_step("普通网络输出 y_pred = F(x)", plain_y)
    print_step("残差网络输出 y_pred = x + F(x)", residual_y)

    print("手算关系：")
    print("hidden = x * 2")
    print("F(x) = hidden * 3 = x * 6")
    print("plain 输出 = 6x")
    print("residual 输出 = x + 6x = 7x")
    print()
    print("plain 和 residual 的 hidden 一样吗:", np.allclose(plain_hidden, residual_hidden))
    print("plain 和 residual 的 F(x) 一样吗:", np.allclose(plain_branch, residual_branch))


if __name__ == "__main__":
    main()
