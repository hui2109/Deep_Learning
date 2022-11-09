import numpy as np
from PIL import Image

from minist数据集.mnist import load_mnist


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True)
img = x_train[0]
print(img)
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
