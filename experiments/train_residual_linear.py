"""从 0 写一个可训练的最小残差模型。

任务：让模型学会 y = 2x。

普通线性模型可以写成：
    y_pred = x * w + b

这里用残差写法：
    F(x) = x * w + b
    y_pred = x + F(x)

如果目标是 y = 2x，最理想的情况就是：
    F(x) 学到 x
    所以 w 接近 1，b 接近 0
"""

import numpy as np


def make_dataset():
    """构造一个非常小的数据集，方便手算和观察。"""
    x = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y = 2 * x
    return x, y


def forward(x, params):
    """前向传播：先算 F(x)，再把原始输入 x 加回来。"""
    branch = x @ params["w"] + params["b"]
    y_pred = x + branch
    return y_pred, branch


def mean_squared_error(y_pred, y):
    """均方误差：预测值和真实值差距越大，loss 越大。"""
    error = y_pred - y
    return np.mean(error**2)


def loss_and_grads(x, y, params):
    """手写反向传播，算 loss 对 w 和 b 的梯度。"""
    y_pred, _ = forward(x, params)
    error = y_pred - y
    loss = np.mean(error**2)

    # loss = mean(error^2)，所以 d_loss/d_y_pred = 2 * error / 样本数。
    grad_y_pred = 2 * error / x.shape[0]

    # y_pred = x + branch，branch = x @ w + b。
    # x 是原始输入，不是可训练参数；真正要更新的是 w 和 b。
    grad_w = x.T @ grad_y_pred
    grad_b = np.sum(grad_y_pred, axis=0, keepdims=True)

    grads = {"w": grad_w, "b": grad_b}
    return loss, grads


def sgd_step(params, grads, learning_rate):
    """最小 SGD：参数往梯度的反方向走一小步。"""
    return {
        "w": params["w"] - learning_rate * grads["w"],
        "b": params["b"] - learning_rate * grads["b"],
    }


def train(x, y, steps=100, learning_rate=0.05):
    """训练残差模型，返回训练后的参数和 loss 记录。"""
    params = {
        "w": np.array([[0.0]]),
        "b": np.array([[0.0]]),
    }
    history = []

    for _ in range(steps):
        loss, grads = loss_and_grads(x, y, params)
        history.append(loss)
        params = sgd_step(params, grads, learning_rate)

    return params, history


def main():
    x, y = make_dataset()
    params, history = train(x, y, steps=80, learning_rate=0.05)
    y_pred, branch = forward(x, params)

    print("任务：让残差模型学会 y = 2x")
    print("模型：y_pred = x + F(x)，其中 F(x) = x @ w + b")
    print()
    print("训练前的直觉：如果 F(x) 学会 x，那么 x + F(x) 就是 2x。")
    print()
    print("训练后的参数：")
    print("w =", params["w"])
    print("b =", params["b"])
    print()
    print("loss 开始:", history[0])
    print("loss 结束:", history[-1])
    print()
    print("x:")
    print(x.T)
    print("真实 y = 2x:")
    print(y.T)
    print("主分支 F(x):")
    print(branch.T)
    print("残差输出 y_pred = x + F(x):")
    print(y_pred.T)


if __name__ == "__main__":
    main()
