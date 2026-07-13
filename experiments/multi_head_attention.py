"""最小两头 Multi-Head Attention 演示。

每个 head 使用自己的 Wq、Wk、Wv；head 输出拼接后，再经 Wo 投影回模型维度。
"""

import numpy as np


def make_demo_inputs():
    """构造 3 个 token 的二维输入 X。"""
    return np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )


def make_demo_params():
    """构造两个一维 head 的固定投影矩阵，以及最终输出矩阵 Wo。"""
    return {
        "heads": [
            {
                "wq": np.array([[1.0], [0.0]]),
                "wk": np.array([[1.0], [0.0]]),
                "wv": np.array([[1.0], [0.0]]),
            },
            {
                "wq": np.array([[0.0], [1.0]]),
                "wk": np.array([[0.0], [1.0]]),
                "wv": np.array([[0.0], [1.0]]),
            },
        ],
        "wo": np.array(
            [
                [1.0, 0.5],
                [-0.5, 1.0],
            ]
        ),
    }


def softmax(scores):
    """按行将 attention scores 转成权重。"""
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def single_head_attention(inputs, head_params):
    """计算一个 attention head。"""
    queries = inputs @ head_params["wq"]
    keys = inputs @ head_params["wk"]
    values = inputs @ head_params["wv"]
    scores = queries @ keys.T / np.sqrt(keys.shape[1])
    weights = softmax(scores)
    outputs = weights @ values
    return queries, keys, values, scores, weights, outputs


def multi_head_attention(inputs, params):
    """计算多个 head，拼接结果并执行最终的输出投影。"""
    head_results = [single_head_attention(inputs, head) for head in params["heads"]]
    head_outputs = [result[-1] for result in head_results]
    concatenated = np.concatenate(head_outputs, axis=1)
    outputs = concatenated @ params["wo"]
    return head_results, concatenated, outputs


def main():
    inputs = make_demo_inputs()
    params = make_demo_params()
    head_results, concatenated, outputs = multi_head_attention(inputs, params)

    print("最小 Two-Head Multi-Head Attention")
    print("X（每一行是一个 token 向量）:\n", inputs)
    print()
    for index, result in enumerate(head_results, start=1):
        queries, keys, values, scores, weights, head_output = result
        print(f"Head {index}")
        print("Q:\n", queries)
        print("K:\n", keys)
        print("V:\n", values)
        print("attention weights:\n", weights)
        print("head output:\n", head_output)
        print()

    print("concatenated = Concat(head_1_output, head_2_output):\n", concatenated)
    print("final output = concatenated @ Wo:\n", outputs)
    print()
    print("关键直觉：")
    print("不同 head 可以学习不同的匹配方式；Concat 保留各 head 的结果，Wo 再把它们混合。")


if __name__ == "__main__":
    main()
