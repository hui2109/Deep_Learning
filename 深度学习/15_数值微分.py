import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 0.01 * x ** 2 + 0.1 * x


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def target_line(f, x):
    k = numerical_diff(f, x)
    b = f(x) - k * x
    return lambda t: k * t + b


x = np.arange(0, 20, 0.1)
y = f(x)
y2 = target_line(f, 10)(x)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
plt.plot(x, y2)
plt.show()
