import numpy as np
import matplotlib.pylab as pl
def softmax1(a):
    exp_a=np.exp(a)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y
a=np.array([1010,1000,990])
# print(softmax1(a))
b=a-np.max(a)
print(softmax1(b))

def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c) # 溢出对策
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y
y=softmax(a)
print(y)
print(np.sum(y))