"""对比普通线性模型和残差线性模型的训练差异。

任务都一样：学习 y = 2x。

普通模型：
    y_pred = F(x)
    F(x) = x @ w + b
    理想结果：w 接近 2，b 接近 0

残差模型：
    y_pred = x + F(x)
    F(x) = x @ w + b
    理想结果：w 接近 1，b 接近 0
"""

import numpy as np


def make_dataset():
    """构造 5 个一维样本，目标是让模型学会 y = 2x。"""
    x = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y = 2 * x
    return x, y


def make_initial_params():
    """初始化一个最小线性层的参数。"""
    return {
        "w": np.array([[0.0]]),
        "b": np.array([[0.0]]),
    }


def linear_branch(x, params):
    """主分支 F(x)：当前只用一层线性变换。"""
    return x @ params["w"] + params["b"]


def plain_forward(x, params):
    """普通模型：输出完全依赖 F(x)。"""
    branch = linear_branch(x, params)
    y_pred = branch
    return y_pred, branch


def residual_forward(x, params):
    """残差模型：输出等于原始输入 x 加上 F(x)。"""
    branch = linear_branch(x, params)
    y_pred = x + branch
    return y_pred, branch


def loss_and_grads(forward_fn, x, y, params):
    """根据传入的 forward 函数，计算 loss 和参数梯度。"""
    y_pred, _ = forward_fn(x, params)
    error = y_pred - y
    loss = np.mean(error**2)

    # loss = mean(error^2)，所以 d_loss/d_y_pred = 2 * error / 样本数。
    grad_y_pred = 2 * error / x.shape[0]

    # plain 和 residual 对 w、b 的梯度公式相同。
    # residual 里的 +x 不含可训练参数，所以不会改变 w、b 的求导路径。
    grad_w = x.T @ grad_y_pred
    grad_b = np.sum(grad_y_pred, axis=0, keepdims=True)

    grads = {"w": grad_w, "b": grad_b}
    return loss, grads


def sgd_step(params, grads, learning_rate):
    """参数沿梯度反方向更新。"""
    return {
        "w": params["w"] - learning_rate * grads["w"],
        "b": params["b"] - learning_rate * grads["b"],
    }


def train(forward_fn, x, y, steps=100, learning_rate=0.05):
    """训练一个指定 forward 形式的线性模型。"""
    params = make_initial_params()
    history = []

    for _ in range(steps):
        loss, grads = loss_and_grads(forward_fn, x, y, params)
        history.append(loss)
        params = sgd_step(params, grads, learning_rate)

    return params, history


def print_result(title, formula, x, y, params, history, forward_fn):
    """打印一组模型的训练结果。"""
    y_pred, branch = forward_fn(x, params)

    print(title)
    print("公式:", formula)
    print("训练后的 w:", params["w"])
    print("训练后的 b:", params["b"])
    print("loss 开始:", history[0])
    print("loss 结束:", history[-1])
    print("F(x):", branch.T)
    print("预测 y_pred:", y_pred.T)
    print("真实 y:", y.T)
    print()


def main():
    x, y = make_dataset()

    plain_params, plain_history = train(plain_forward, x, y, steps=100, learning_rate=0.05)
    residual_params, residual_history = train(
        residual_forward, x, y, steps=100, learning_rate=0.05
    )

    print("目标：学习 y = 2x")
    print("对照重点：两者都用 F(x) = x @ w + b，只差有没有把 x 加回来。")
    print()
    print("x:", x.T)
    print("真实 y:", y.T)
    print()

    print_result(
        "普通线性模型",
        "y_pred = F(x)",
        x,
        y,
        plain_params,
        plain_history,
        plain_forward,
    )
    print_result(
        "残差线性模型",
        "y_pred = x + F(x)",
        x,
        y,
        residual_params,
        residual_history,
        residual_forward,
    )


if __name__ == "__main__":
    main()
