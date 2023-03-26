import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Relu(x):
    mask = (x <= 0)
    out = x.copy()
    out[mask] = 0
    return out


x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]

    # w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.randn(node_num, node_num) * np.sqrt(1/node_num)

    # w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.randn(node_num, node_num) * np.sqrt(1/node_num)
    w = np.random.randn(node_num, node_num) * np.sqrt(2/node_num)

    z = np.dot(x, w)
    # a = sigmoid(z)
    a = Relu(z)
    activations[i] = a

for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + '-layer')
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()
