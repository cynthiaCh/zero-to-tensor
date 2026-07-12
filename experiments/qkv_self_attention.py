"""最小 Q/K/V self-attention 演示。

同一组 token 向量 X 分别乘以 Wq、Wk、Wv，得到 Q、K、V，
再用 Q 和 K 决定从各位置取多少 V。
"""

import numpy as np


def make_demo_inputs():
    """构造 3 个 token 的二维表示 X。"""
    return np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )


def make_demo_params():
    """构造固定的 Q、K、V 投影矩阵。真实模型会在训练中学习它们。"""
    return {
        "wq": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "wk": np.array([[1.0, 1.0], [0.0, 1.0]]),
        "wv": np.array([[1.0, 0.0], [1.0, 1.0]]),
    }


def softmax(scores):
    """按行将 attention scores 转成概率权重。"""
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def project_qkv(inputs, params):
    """让同一个输入 X 分别得到 Q、K、V 三种表示。"""
    queries = inputs @ params["wq"]
    keys = inputs @ params["wk"]
    values = inputs @ params["wv"]
    return queries, keys, values


def self_attention(inputs, params):
    """计算所有 token 同时对所有 token 的 self-attention 输出。"""
    queries, keys, values = project_qkv(inputs, params)
    scores = queries @ keys.T / np.sqrt(keys.shape[1])
    weights = softmax(scores)
    outputs = weights @ values
    return queries, keys, values, scores, weights, outputs


def main():
    inputs = make_demo_inputs()
    params = make_demo_params()
    queries, keys, values, scores, weights, outputs = self_attention(inputs, params)

    print("最小 Q/K/V Self-Attention")
    print("X（每一行是一个 token 向量）:\n", inputs)
    print()
    print("Q = X @ Wq:\n", queries)
    print("K = X @ Wk:\n", keys)
    print("V = X @ Wv:\n", values)
    print()
    print("scores = Q @ K.T / sqrt(key_dimension):\n", scores)
    print("weights = softmax(scores)，每一行的和:", weights.sum(axis=1))
    print(weights)
    print("outputs = weights @ V:\n", outputs)
    print()
    print("关键直觉：")
    print("每一行 Q 是“当前位置要找什么”；每一行 K 是“当前位置可被怎样匹配”。")
    print("V 是真正被取回的信息。Q 与 K 决定权重，权重再对 V 做加权平均。")


if __name__ == "__main__":
    main()
