import math

import matplotlib.pyplot as plt


def draw_sqrt2_triangle():
    """画出两条直角边为 1 的等腰直角三角形。"""
    points_x = [0, 1, 0, 0]
    points_y = [0, 0, 1, 0]
    hypotenuse = math.sqrt(2)

    plt.figure(figsize=(6, 6))
    plt.plot(points_x, points_y, marker="o", linewidth=2)

    # 用等比例坐标展示，避免视觉上把直角三角形拉伸变形。
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.title("Right Isosceles Triangle: hypotenuse = sqrt(2)")

    plt.text(0.5, -0.08, "1", ha="center", va="top")
    plt.text(-0.08, 0.5, "1", ha="right", va="center")
    plt.text(0.55, 0.55, f"sqrt(2) = {hypotenuse:.12f}", ha="left", va="bottom")

    plt.xlim(-0.2, 1.3)
    plt.ylim(-0.2, 1.3)
    plt.show()


if __name__ == "__main__":
    draw_sqrt2_triangle()

