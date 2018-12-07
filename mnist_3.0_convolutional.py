# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import math
from matplotlib import pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data as mnist_data
tf.set_random_seed(0)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1=>4 stride 1        W1 [5, 5, 1, 4]        B1 [4]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 4]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x4=>8 stride 2        W2 [5, 5, 4, 8]        B2 [8]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 8]
#     @ @ @ @ @ @       -- conv. layer 4x4x8=>12 stride 2       W3 [4, 4, 8, 12]       B3 [12]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 12] => reshaped to YY [batch, 7*7*12]
#      \x/x\x\x/        -- fully connected layer (relu)         W4 [7*7*12, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]           B5 [10]
#        · · ·                                                  Y [batch, 10]

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
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
stride = 1  # output is 28x28 [100, 28, 28, 4]  relu: x<=0 -> y=0, x>0 -> y = x
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2
#out size = (N - F) / strides + 1  (28 - 5) / 2 + 1 = 13
# output is 14x14 [100, 28, 28, 8]   (28 - 2)/ 2 + 1 = 14  过滤器移动的步长，第一位和第四位一般恒定为1，第二位指水平移动时候的步长，第三位指垂直移动的步长。strides = [1, stride, stride, 1].
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7g (14 - 2)/ 2 + 1 =7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
# Y3 (100, 7, 7, 16) reshape-> (100, 784)
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

# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)


saver = tf.train.Saver() 

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# You can call this function in a loop to train the model, 100 images at a time
for i in range(2000):
    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: 0.003})
    # compute training values for visualisation
    if i % 500 == 0:
        a, c, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases], {X: batch_X, Y_: batch_Y})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
        Y1_OUT, Y2_OUT, Y3_OUT, Y4_OUT, W1_OUT, W2_OUT, YY_OUT = sess.run([Y1, Y2, Y3, Y4, W1, W2, YY], {X: batch_X, Y_: batch_Y})
        print(Y1_OUT.shape)
        print(Y2_OUT.shape)
        print(Y3_OUT.shape)
        print(Y4_OUT.shape)
    if i % 1000 == 0:
        W4_OUT = sess.run(W4, {X: batch_X, Y_: batch_Y})
        print(W4_OUT.shape)
        temp = [];
        for k, v in enumerate(W4_OUT):
            temp.append(v[9]);
        temp = np.array(temp)
        temp = temp.reshape(28, 28) #不能绘制 
        #print(temp)
        plt.imshow(temp, cmap='magma') #, cmap="inferno"
        plt.colorbar() 
        #plt.show()
saver.save(sess, "testdata2/model.ckpt") 

"""   
if i % 1000 == 0:
    Y3_OUT = sess.run(Y3, {X: batch_X, Y_: batch_Y})
    print(Y3_OUT)
    temp = tf.reshape(Y3_OUT[0], [28, 28])
    temp = temp.eval(session=sess)
    plt.imshow(temp) #, cmap="inferno"
    plt.show()
"""
    
"""
#转化为numpy数组
img_numpy=img.eval(session=sess)
print("out2=",type(img_numpy))
#转化为tensor
img_tensor= tf.convert_to_tensor(img_numpy)
print("out2=",type(img_tensor))
"""
# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)

# layers 4 8 12 200, patches 5x5str1 5x5str2 4x4str2 best 0.989 after 10000 iterations
# layers 4 8 12 200, patches 5x5str1 4x4str2 4x4str2 best 0.9892 after 10000 iterations
# layers 6 12 24 200, patches 5x5str1 4x4str2 4x4str2 best 0.9908 after 10000 iterations but going downhill from 5000 on
# layers 6 12 24 200, patches 5x5str1 4x4str2 4x4str2 dropout=0.75 best 0.9922 after 10000 iterations (but above 0.99 after 1400 iterations only)
# layers 4 8 12 200, patches 5x5str1 4x4str2 4x4str2 dropout=0.75, best 0.9914 at 13700 iterations
# layers 9 16 25 200, patches 5x5str1 4x4str2 4x4str2 dropout=0.75, best 0.9918 at 10500 (but 0.99 at 1500 iterations already, 0.9915 at 5800)
# layers 9 16 25 300, patches 5x5str1 4x4str2 4x4str2 dropout=0.75, best 0.9916 at 5500 iterations (but 0.9903 at 1200 iterations already)
# attempts with 2 fully-connected layers: no better 300 and 100 neurons, dropout 0.75 and 0.5, 6x6 5x5 4x4 patches no better
#*layers 6 12 24 200, patches 6x6str1 5x5str2 4x4str2 dropout=0.75 best 0.9928 after 12800 iterations (but consistently above 0.99 after 1300 iterations only, 0.9916 at 2300 iterations, 0.9921 at 5600, 0.9925 at 20000)
# layers 6 12 24 200, patches 6x6str1 5x5str2 4x4str2 no dropout best 0.9906 after 3100 iterations (avove 0.99 from iteration 1400)
