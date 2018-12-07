
import tensorflow as tf
import math
from matplotlib import pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data as mnist_data
tf.set_random_seed(0)

mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# variable learning rate
lr = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 16  # third convolutional layer 12
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2
#out size = (N - F) / strides + 1  (28 - 5) / 2 + 1 = 13
# output is 14x14   (28 - 2)/ 2 + 1 = 14  过滤器移动的步长，第一位和第四位一般恒定为1，第二位指水平移动时候的步长，第三位指垂直移动的步长。strides = [1, stride, stride, 1].
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7g
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
# Y3 (100, 7, 7, 12) reshape-> (100, 588)
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])
# full connect first YY * W4  (100, 588) * (588, 200) = (100, 200)
Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
# full connect second Y4 * W5  (100, 200) * (200, 10) = （100， 10）将每一张图片会连接为10列，10个数字，将来会强化一个最明显的特征，即要得出的结论
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits) # （100， 10）加起来=100%

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

saver = tf.train.Saver() 
# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_x,  batch_y = mnist.train.next_batch(60)

saver.restore(sess, "testdata2/model.ckpt")  
prediction = tf.argmax(Y, 1)
predint = prediction.eval(feed_dict={X: batch_x}, session=sess)

orgin = tf.argmax(batch_y, 1)
print(sess.run(orgin))
print(predint)
