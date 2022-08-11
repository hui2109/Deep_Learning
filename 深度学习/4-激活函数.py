import numpy as np
import matplotlib.pylab as pl
def step_function1(x):
    if x>0:
        return 1
    else:
        return 0    
def step_function2(x):
    y=x>0
    return y.astype(int)
def step_function3(x):
    return np.array(x>0,dtype=int)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0,x)

# x=np.arange(-5.0,5.0,0.1)
# y=step_function3(x)
# pl.plot(x,y)
# pl.ylim(-0.1,1.1)
# pl.show()

# x=np.arange(-5.0,5.0,0.1)
# y=sigmoid(x)
# pl.plot(x,y)
# pl.ylim(-0.1,1.1)
# pl.show()

x=np.arange(-5.0,5.0,0.1)
y=relu(x)
pl.plot(x,y)
pl.ylim(-0.1,5.1)
pl.show()