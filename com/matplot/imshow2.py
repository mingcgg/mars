from matplotlib import pyplot as plt
from matplotlib import animation
import time
import numpy as np

fig = plt.figure()

xx = 2001 // 20
print(xx)
ax0 = None
ax1 = None
img0 = None
img1 = None
def _init():
    plot0 = fig.add_subplot(2, 2, 1)
    img0 = np.random.random(size = (28, 28))
    plot1 = fig.add_subplot(2, 2, 2)
    img1 = np.random.random(size = (28, 28))
    #引用全局变量，不需要golbal声明，修改全局变量，需要使用global声明，特别地，列表、字典等如果只是修改其中元素的值，可以直接使用全局变量，不需要global声明。
    global ax0
    ax0 = plot0.imshow(img0, cmap="jet") 
    global ax1
    ax1 = plot1.imshow(img1, cmap="jet") 
    return ax0, ax1

def animate(i):
    print(i)
    ax0.set_data(img0)
    ax1.set_data(img1)
    return img0

def updateImg():
    global img0
    img0 = np.random.random(size = (28, 28))
    global img1
    img1 = np.random.random(size = (28, 28))

for i in range(200):
    updateImg()
    #time.sleep(1)

ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=1000,
                              init_func=_init,
                              interval=100,
                              blit=False)
plt.show()

