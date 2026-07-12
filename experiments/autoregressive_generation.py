"""最小 auto-regressive 文本生成演示。

模型每次根据当前前缀预测一个下一个 token，再把该 token 加回前缀，继续预测。
这里用固定的转移 logits 代替训练好的 Transformer，专门展示生成循环。
"""

import numpy as np


VOCABULARY = ("<BOS>", "我", "爱", "学习", "<EOS>")
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCABULARY)}


def make_transition_logits():
    """构造一个确定的 toy 模型：<BOS> -> 我 -> 爱 -> 学习 -> <EOS>。"""
    low_score = -20.0
    return np.array(
        [
            [low_score, 4.0, 0.0, 0.0, low_score],
            [low_score, low_score, 4.0, 0.0, low_score],
            [low_score, low_score, low_score, 4.0, 0.0],
            [low_score, low_score, low_score, low_score, 4.0],
            [low_score, low_score, low_score, low_score, 4.0],
        ]
    )


def softmax(logits):
    """将 logits 转成概率分布。"""
    shifted_logits = logits - np.max(logits)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / exp_logits.sum()


def predict_next_token(prefix, transition_logits):
    """根据当前前缀的最后一个 token 给出下一个 token 的概率。"""
    previous_token = prefix[-1]
    logits = transition_logits[TOKEN_TO_ID[previous_token]]
    probabilities = softmax(logits)
    next_token = VOCABULARY[np.argmax(probabilities)]
    return next_token, probabilities


def generate(transition_logits, max_new_tokens=10):
    """自回归生成：每一步都将刚生成的 token 接到 prefix 后面。"""
    prefix = ["<BOS>"]
    steps = []

    for _ in range(max_new_tokens):
        next_token, probabilities = predict_next_token(prefix, transition_logits)
        steps.append(
            {
                "prefix": prefix.copy(),
                "next_token": next_token,
                "probabilities": probabilities,
            }
        )
        prefix.append(next_token)
        if next_token == "<EOS>":
            break

    return prefix, steps


def main():
    transition_logits = make_transition_logits()
    generated_tokens, steps = generate(transition_logits)

    print("最小 Auto-Regressive 生成")
    print("固定规则：<BOS> -> 我 -> 爱 -> 学习 -> <EOS>")
    print()
    for index, step in enumerate(steps, start=1):
        best_probability = step["probabilities"].max()
        print(f"第 {index} 步")
        print("当前 prefix:", " ".join(step["prefix"]))
        print(f"预测 next token: {step['next_token']}（概率 {best_probability:.4f}）")
        print()

    print("最终序列:", " ".join(generated_tokens))
    print("关键直觉：每次生成的结果都会成为下一次输入的一部分。")
    print("Transformer decoder 用 causal mask 保证每个位置只能注意到自己和过去的位置。")


if __name__ == "__main__":
    main()
