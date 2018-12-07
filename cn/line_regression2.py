#======非线性回归 tensorboard mine =========
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200,dtype=np.float32)[:,np.newaxis]
#print(x_data.shape)
noise = np.random.normal(0, 0.02, x_data.shape)

y_data = np.square(x_data) + noise
#print(y_data)
#定义两个placeholder
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

#定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1

L1 = tf.nn.tanh(Wx_plus_b_L1)

#定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2

prediction = tf.nn.tanh(Wx_plus_b_L2)

with tf.name_scope('layerOut'):
    tf.summary.histogram('L2/Weihts', Weights_L2)

#二次代价函数 
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y - prediction))
    tf.summary.scalar('loss', loss)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
merge = tf.summary.merge_all()
writer = tf.summary.FileWriter('Desktop/outputs2/', sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(1000):
    _, result,LOSS = sess.run([train_step, merge, loss], feed_dict={x: x_data, y: y_data})
    if i % 100 == 0:
        print(LOSS)
        tf.reset_default_graph()
        writer.add_summary(result, i)
    
prediction_value = sess.run(prediction, feed_dict={x: x_data})
Weights_L1_value = sess.run(Weights_L1, feed_dict={x: x_data})
L1_value = sess.run(L1, feed_dict={x: x_data})

#draw
plt.figure()
plt.scatter(x_data, y_data)
plt.plot(x_data, prediction_value, 'r-', lw=1)
plt.show()