import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, _x):
        return np.dot(_x, self.W)

    def loss(self, _x, _t):
        z = self.predict(_x)
        y = softmax(z)
        loss = cross_entropy_error(y, _t)
        return loss


if __name__ == '__main__':
    net = SimpleNet()
    print(net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))

    t = np.array([0, 0, 1])
    print(net.loss(x, t))

    result = numerical_gradient(lambda w: net.loss(x, t), net.W)
    print(result)
