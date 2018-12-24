import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from matplotlib import animation
from matplotlib import pyplot as plt

lrate = 0.005

#def create_graph():
Y_ = tf.placeholder(tf.float32, shape=[None, 10])

X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))

X_reshape = tf.reshape(X, shape=[-1, 784])
logits = tf.matmul(X_reshape, weights) + biases
Y = tf.nn.softmax(logits)
loss = -(Y_ * tf.log(Y)) * 1000
cross_entropy = tf.reduce_mean(loss)
train_step = tf.train.GradientDescentOptimizer(lrate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

mnist = mnist_data.read_data_sets("../data", one_hot=True, reshape=False, validation_size=0)

fig = plt.figure()
axs = []
imgs = []   
def init_mpl():
    global axs
    global imgs
    for i in range(10):
        plot = fig.add_subplot(2, 5, i + 1)
        img = np.random.random(size = (28, 28))
        ax = plot.imshow(img, animated=True)  #, cmap='jet', vmin=0.0, vmax=1.0, interpolation='nearest', aspect=1.0
        imgs.append(img) 
        axs.append(ax)
    return axs, imgs
def training_step(i):
    global imgs
    #plt.pause(0.01)
    batch_x,  batch_y = mnist.train.next_batch(10)
    sess.run(train_step, feed_dict={X:batch_x, Y_:batch_y}) 
    _,accuracy_out, WW, cross_entropy_out = sess.run([train_step,accuracy, weights, cross_entropy], feed_dict={X:batch_x, Y_:batch_y})
    #if i % 500 == 0:
    print("accuracy:", accuracy_out)
    print("loss:", cross_entropy_out)
    # update imgs
    ws = np.hsplit(WW, 10) # 垂直分割
    ws = np.squeeze(ws)
    for n in range(10):
        array = np.array(ws[n], dtype=float)
        imgs[n] = array.reshape(28, 28)
def animate(i):
    global axs
    training_step(i)
    # update axs
    for n in range(10):
        axs[n].set_data(imgs[n])
    return axs
        
ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=1000,
                              init_func=init_mpl,
                              interval=100,
                              blit=False)
        
plt.show()
