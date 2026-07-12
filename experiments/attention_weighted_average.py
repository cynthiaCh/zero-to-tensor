"""最小 scaled dot-product attention 前向传播演示。

不训练参数，只用固定的 query、key、value 看清楚注意力怎样从多个位置
取回信息，并将它们做成一个加权平均。
"""

import numpy as np


def make_demo_inputs():
    """构造一个 query、三个 key 和三个来自不同位置的 value。"""
    query = np.array([[1.0, 0.0]])
    keys = np.array(
        [
            [2.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    values = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
        ]
    )
    return query, keys, values


def softmax(scores):
    """将分数变成和为 1 的权重。减最大值是数值稳定处理。"""
    shifted_scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


def scaled_dot_product_attention(query, keys, values):
    """计算一个 query 对所有位置的注意力输出。"""
    key_dimension = keys.shape[1]
    scores = query @ keys.T / np.sqrt(key_dimension)
    weights = softmax(scores)
    context = weights @ values
    return scores, weights, context


def main():
    query, keys, values = make_demo_inputs()
    scores, weights, context = scaled_dot_product_attention(query, keys, values)

    print("最小 Transformer Attention 前向传播")
    print("query:", query)
    print("keys:\n", keys)
    print("values（每一行来自序列的一个位置）:\n", values)
    print()
    print("1. score = query @ keys.T / sqrt(key_dimension)")
    print(scores)
    print("2. weights = softmax(score)，权重和为:", weights.sum())
    print(weights)
    print("3. context = weights @ values")
    print(context)
    print()
    print("逐位置展开：")
    for index, (weight, value) in enumerate(zip(weights[0], values)):
        print(f"位置 {index}: {weight:.4f} * {value} = {weight * value}")
    print()
    print("关键直觉：")
    print("query 与 key 越相似，该位置的 weight 越大。")
    print("最终 context 是所有位置 value 的加权平均，所以模型能按相关性汇总序列信息。")


if __name__ == "__main__":
    main()
