"""最小 Transformer position-wise feed-forward network 演示。

FFN 不在 token 位置之间传递信息；每一行都独立经过同一组两层参数。
"""

import numpy as np


def make_demo_inputs():
    """构造 3 个位置的二维向量。"""
    return np.array(
        [
            [1.0, 2.0],
            [-1.0, 3.0],
            [2.0, -2.0],
        ]
    )


def make_demo_params():
    """构造 2 -> 3 -> 2 的固定 FFN 参数。"""
    return {
        "w1": np.array(
            [
                [1.0, -1.0, 0.5],
                [0.5, 1.0, -0.5],
            ]
        ),
        "b1": np.array([0.0, -0.5, 0.0]),
        "w2": np.array(
            [
                [1.0, 0.5],
                [-1.0, 1.0],
                [0.5, -0.5],
            ]
        ),
        "b2": np.array([0.1, -0.2]),
    }


def relu(values):
    """ReLU 将负数变成 0。"""
    return np.maximum(values, 0.0)


def feed_forward(inputs, params):
    """对每个位置应用同一组两层网络。"""
    hidden_raw = inputs @ params["w1"] + params["b1"]
    hidden = relu(hidden_raw)
    outputs = hidden @ params["w2"] + params["b2"]
    return hidden_raw, hidden, outputs


def main():
    inputs = make_demo_inputs()
    params = make_demo_params()
    hidden_raw, hidden, outputs = feed_forward(inputs, params)

    print("最小 Transformer Feed-Forward Network")
    print("inputs（每一行对应一个 token 位置）:\n", inputs)
    print()
    print("1. hidden_raw = inputs @ w1 + b1")
    print(hidden_raw)
    print("2. hidden = ReLU(hidden_raw)")
    print(hidden)
    print("3. outputs = hidden @ w2 + b2")
    print(outputs)
    print()
    print("逐位置结果：")
    for index, (input_row, output_row) in enumerate(zip(inputs, outputs)):
        print(f"位置 {index}: {input_row} -> {output_row}")
    print()
    print("关键直觉：")
    print("每个位置都使用同样的 w1、b1、w2、b2，但位置之间没有在 FFN 内直接相互读取。")
    print("Attention 负责跨位置取信息；FFN 负责对每个位置取回的信息做非线性加工。")


if __name__ == "__main__":
    main()
