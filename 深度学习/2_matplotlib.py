import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
# 生成数据
x=np.arange(0,6,0.1)
y1=np.sin(x)
y2=np.cos(x)

# 绘制图形
plt.plot(x,y1,label='sin')
plt.plot(x,y2,linestyle='--',label='cos')
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin & cos')
plt.legend()
plt.show()

img=imread(r"C:\Users\99563\Pictures\02. 峨眉山 2018-2-1\IMG20180130145736.jpg")
plt.imshow(img)
plt.show()