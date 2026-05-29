import numpy as np
import matplotlib.pyplot as plt


def draw_exponential_waiting_time():
    """画出平均每小时 1 次事件时，等待时间的指数分布。"""
    rate_per_hour = 1
    minutes = np.linspace(0, 180, 400)
    hours = minutes / 60
    density = rate_per_hour * np.exp(-rate_per_hour * hours)

    plt.figure(figsize=(8, 5))
    plt.plot(minutes, density, linewidth=2, label="waiting time density")

    # 第 1 分钟附近的密度最高，但这不是说钟表上的第 1 分钟更危险。
    plt.scatter([1], [rate_per_hour * np.exp(-rate_per_hour / 60)], color="red", zorder=3)
    plt.text(4, 0.98, "near the first minute", color="red")

    plt.title("Exponential Waiting Time: average 1 event per hour")
    plt.xlabel("Waiting time from now (minutes)")
    plt.ylabel("Probability density")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    draw_exponential_waiting_time()

