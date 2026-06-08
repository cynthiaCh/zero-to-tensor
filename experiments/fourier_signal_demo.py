"""傅里叶变换最小例子：从一段信号里找出主要频率。

AI 里的很多数据不是只看原始值，也会先换一种表示。
傅里叶变换做的事就是把“时间上的波动”换成“频率上的强弱”。
"""

import numpy as np


def make_mixed_signal(sample_count=128):
    """构造一个由两个正弦波混合成的信号。"""
    t = np.arange(sample_count) / sample_count

    # 这个信号故意由频率 3 和频率 15 组成。
    low_frequency = np.sin(2 * np.pi * 3 * t)
    high_frequency = 0.5 * np.sin(2 * np.pi * 15 * t)
    return low_frequency + high_frequency


def spectrum_strength(signal):
    """用 FFT 把信号从时间域转到频率域。"""
    sample_count = signal.shape[0]
    fft_values = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(sample_count, d=1 / sample_count)

    # abs 表示每个频率成分的强度；除以样本数让数值更容易比较。
    strengths = np.abs(fft_values) / sample_count
    return frequencies, strengths


def find_top_frequencies(frequencies, strengths, count=3):
    """找出强度最高的几个非零频率。"""
    non_zero_indexes = np.where(frequencies > 0)[0]
    sorted_indexes = non_zero_indexes[np.argsort(strengths[non_zero_indexes])[::-1]]

    top = []
    for index in sorted_indexes[:count]:
        top.append(
            {
                "frequency": int(round(frequencies[index])),
                "strength": strengths[index],
            }
        )
    return top


def main():
    signal = make_mixed_signal()
    frequencies, strengths = spectrum_strength(signal)
    top = find_top_frequencies(frequencies, strengths, count=2)

    print("傅里叶变换最小例子")
    print()
    print("原始信号前 10 个点：")
    print(signal[:10])
    print()
    print("最强的两个频率：")
    for item in top:
        print(f"频率 {item['frequency']}: 强度 {item['strength']:.4f}")
    print()
    print("这个信号是人为构造的：sin(2π*3*t) + 0.5*sin(2π*15*t)")
    print("所以 FFT 应该能找出频率 3 和频率 15。")
    print()
    print("AI 里的直觉：")
    print("原始信号是一串数字；FFT 后，模型可以直接看到哪些频率最明显。")


if __name__ == "__main__":
    main()
