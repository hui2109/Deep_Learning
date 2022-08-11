import numpy as np
import matplotlib.pylab as pl
a=np.array([1,2,3,4])
print(a)
print(np.ndim(a))
print(a.shape)
print(a.shape[0])
print('------------------------------------------------')
b=np.array([[1,2,3],[3,4,5],[5,6,7],[8,9,10]])
print(b)
print(np.ndim(b))
print(b.shape)
print('------------------------------------------------')
c=np.array([[[1,2,3],[3,4,5],[5,6,7],[8,9,10]],
            [[1,2,3],[3,4,5],[5,6,7],[8,9,10]]])
print(c)
print(np.ndim(c))
print(c.shape)
print('------------------------------------------------')
d=np.array([[[[1,2,3],[3,4,5],[5,6,7],[8,9,10]],
            [[1,2,3],[3,4,5],[5,6,7],[8,9,10]]],
            [[[1,2,3],[3,4,5],[5,6,7],[8,9,10]],
            [[1,2,3],[3,4,5],[5,6,7],[8,9,10]]]])
print(d)
print(np.ndim(d))
print(d.shape)