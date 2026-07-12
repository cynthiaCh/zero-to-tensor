"""最小 RNN 前向传播演示。

这一步先不训练，只看一个序列如何按时间步更新 hidden state。
"""

import numpy as np


def make_demo_sequence():
    """构造 3 个时间步、每步 1 个特征的输入序列。"""
    return np.array([[1.0], [2.0], [3.0]])


def make_demo_params():
    """构造固定参数，让每一步都能跟着公式手算。"""
    return {
        "w_xh": np.array([[0.5]]),
        "w_hh": np.array([[0.1]]),
        "b_h": np.array([[0.0]]),
        "w_hy": np.array([[2.0]]),
        "b_y": np.array([[0.0]]),
    }


def rnn_step(x_t, h_prev, params):
    """单个时间步：用当前输入 x_t 和旧记忆 h_prev 算出新记忆 h_t。"""
    hidden_raw = x_t @ params["w_xh"] + h_prev @ params["w_hh"] + params["b_h"]
    h_t = np.tanh(hidden_raw)
    y_t = h_t @ params["w_hy"] + params["b_y"]
    return h_t, y_t, hidden_raw


def rnn_forward(sequence, params, h0=None):
    """按时间顺序处理整个序列，并保留每一步的中间结果。"""
    if h0 is None:
        h_prev = np.zeros((1, params["w_hh"].shape[0]))
    else:
        h_prev = h0

    steps = []
    for time_index, x_t in enumerate(sequence):
        x_row = x_t.reshape(1, -1)
        h_t, y_t, hidden_raw = rnn_step(x_row, h_prev, params)
        steps.append(
            {
                "time_index": time_index,
                "x_t": x_row,
                "h_prev": h_prev,
                "hidden_raw": hidden_raw,
                "h_t": h_t,
                "y_t": y_t,
            }
        )
        h_prev = h_t

    return steps


def print_step(step):
    """打印一个时间步的完整计算链路。"""
    print(f"t = {step['time_index']}")
    print("x_t:", step["x_t"])
    print("h_prev:", step["h_prev"])
    print("hidden_raw = x_t @ w_xh + h_prev @ w_hh + b_h")
    print(step["hidden_raw"])
    print("h_t = tanh(hidden_raw)")
    print(step["h_t"])
    print("y_t = h_t @ w_hy + b_y")
    print(step["y_t"])
    print()


def main():
    sequence = make_demo_sequence()
    params = make_demo_params()
    steps = rnn_forward(sequence, params)

    print("最小 RNN 前向传播")
    print("输入序列：x = [1, 2, 3]")
    print("参数：w_xh=0.5, w_hh=0.1, b_h=0, w_hy=2, b_y=0")
    print()
    print("核心公式：")
    print("h_t = tanh(x_t @ w_xh + h_{t-1} @ w_hh + b_h)")
    print("y_t = h_t @ w_hy + b_y")
    print()

    for step in steps:
        print_step(step)

    print("关键直觉：")
    print("普通全连接层只看当前 x。")
    print("RNN 每一步还会把上一步的 h_prev 带进来，所以 h_t 同时包含当前输入和过去记忆。")


if __name__ == "__main__":
    main()
