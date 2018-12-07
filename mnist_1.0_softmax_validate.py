import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from matplotlib import pyplot as plt

mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
#from keras.datasets import mnist
#label
Y_ = tf.placeholder(tf.float32,[None, 10])
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
XX = tf.reshape(X, [-1, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

saver = tf.train.Saver() 

sess = tf.Session()

batch_x,  batch_y = mnist.train.next_batch(20)
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
x_train = x_train.reshape(60000, 784)
x = x_train[4].reshape(28, 28)
x = x.reshape(1, 28, 28, 1)
print(x.shape)
y = y_train[4]
print(y)
"""
print(batch_x.shape)
print(batch_y.shape)
def getLabel(data):
    temp = []
    for i in data:
        for idx in range(0,10):
            j = i[idx]
            if j == 1:
                temp.append(idx)
                break
    return np.array(temp)
#print(getLabel(batch_y))
orgin = tf.argmax(batch_y, 1) # 按照某个维度，值最大的索引号
print(sess.run(orgin))
#plt.imshow(x, cmap='magma') #, cmap="inferno"
#plt.colorbar() 
#plt.show()
saver.restore(sess, "testdata/model.ckpt")  
prediction = tf.argmax(Y, 1)
predint = prediction.eval(feed_dict={X: batch_x}, session=sess)

print(predint)

prediction_eq = tf.equal(orgin, prediction)
accuracy = tf.reduce_mean(tf.cast(prediction_eq, tf.float32))
print('accuracy : ', sess.run(accuracy, feed_dict={X: batch_x}))
"""

   
    
"""
