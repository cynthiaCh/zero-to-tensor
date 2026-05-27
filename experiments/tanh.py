import numpy as np  # 数学计算库
import matplotlib.pyplot as plt  # 画图库

x = np.linspace(-3, 3, 100)  # 生成-3到3的100个点

# 直接用你学过的公式定义双曲函数
sinh_x = (np.exp(x) - np.exp(-x)) / 2
cosh_x = (np.exp(x) + np.exp(-x)) / 2
tanh_x = sinh_x / cosh_x

# 画图
plt.plot(x, tanh_x, label='tanh(x)', linewidth=2)
plt.axhline(1, color='gray', linestyle='--')  # 画y=1的渐近线
plt.axhline(-1, color='gray', linestyle='--') # 画y=-1的渐近线
plt.title('Hyperbolic Tangent (tanh)') # 你亲手画的tanh
plt.legend()
plt.grid(True)
plt.show()