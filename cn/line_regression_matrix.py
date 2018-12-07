import tensorflow as tf 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# 一元函数拟合多个权值是没有意义的
# y = x ** 2 + 4
label_x = np.linspace(-1, 1, 200)
noise = np.random.random(200) / 10
label_y = np.square(label_x) + noise #0.4

label_xx = np.reshape(label_x, [200, 1])
label_yy = np.reshape(label_y, [200, 1])

x = tf.placeholder(tf.float32, [200, 1])
y = tf.placeholder(tf.float32, [200, 1])

weights = tf.Variable(tf.random_normal([1, 10]))
bias = tf.Variable(tf.zeros([1, 10]))
Y1 = tf.matmul(x, weights) + bias
L1 = tf.nn.tanh(Y1)

weights2 = tf.Variable(tf.random_normal([10, 1]))
bias2 = tf.Variable(tf.zeros([1, 1], tf.float32))

# x(200, 1) * w(1, 200) = 200 * 200   (200, 1)

prediction = tf.nn.tanh(tf.matmul(L1, weights2) + bias2)
 
loss = tf.reduce_mean(tf.square(y - prediction))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session();
init = tf.global_variables_initializer()
sess.run(init)

def train_step(i):
    _, LOSS, PREDICTION = sess.run([train, loss, prediction], feed_dict={x: label_xx, y:label_yy})
    if i % 10 == 0:
        pp = np.reshape(PREDICTION, [1, 200])
        line.set_ydata(pp)
        print(LOSS)


fig, ax = plt.subplots()
dot, = ax.plot(label_x, label_y, 'ro')
line, = ax.plot(label_x, label_y, 'g--', lw=1)


def animate(i):
    train_step(i)
    return dot

def init():
    return dot

ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=1000,
                              init_func=init,
                              interval=20,
                              blit=False)
plt.show()

# 为什么此示例用sigmoid 及其它激活函数 不行 TODO 20181207 wm