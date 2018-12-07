import tensorflow as tf 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# 计算一个W和一个B的情况，拟合到一条直线上
# 训练，将批量样本与实际结果之间的的差取最小
# y = 0.2x + 0.4 直线
label_x = np.random.random(20)
print(label_x)
noise = np.random.normal(0, 0.01, label_x.shape)
label_y = 0.2 * label_x + noise #0.4

weight = tf.Variable(0.)
bias = tf.Variable(0.)

y = weight * label_x + bias

loss = tf.reduce_mean(tf.square(label_y - y)) #square abs

train = tf.train.AdamOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

fig, ax = plt.subplots()
dot, = ax.plot(label_x, label_y, 'ro')
yy = 0 * label_x + 0
line, = ax.plot(label_x, yy, 'g--')

def train_step(i):
    _, W, B, LOSS = sess.run([train, weight, bias, loss])
    line_y = W * label_x + B
    line.set_ydata(line_y)
    print(i, W, B, LOSS)
    
def animate(i):
    train_step(i)
    return line

def init():
    return line, dot

ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=1000,
                              init_func=init,
                              interval=20,
                              blit=False)
plt.show()