import numpy as np


def function_2(x):
    return np.sum(x ** 2)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad


def gradient_descent(f, _init_x, lr=0.01, step_num=100):
    x = _init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


if __name__ == '__main__':
    init_x = np.array([-3.0, 4.0])
    result = gradient_descent(function_2, _init_x=init_x, lr=0.1, step_num=100)
    print(result)
