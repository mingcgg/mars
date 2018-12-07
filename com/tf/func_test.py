import tensorflow as tf
import numpy as np

# tf.argmax tf.equal

# pic pix number
m2 = np.array([[0, 0, 0, 0, 0.14917259, 0.210407, 0.669037, 0.8345183, 0, 0],
           [1, 2, 7, 4, 5, 8, 6, 3, 9, 0],
           [1, 2, 7, 4, 5, 8, 6, 3, 9, 0]], dtype=np.float32)
Y = tf.nn.softmax(m2)

logY = tf.log(Y) 

Y_ = np.array([[1, 2, 7, 4, 5, 8, 6, 3, 9, 0],
              [1, 2, 7, 4, 5, 8, 6, 3, 9, 0],
              [1, 2, 7, 4, 5, 8, 6, 3, 9, 0]], dtype=np.int)

multi = Y_ * logY # not matrix multi

data1 = np.array([[1, 2, 7, 4, 5, 8, 6, 3, 9, 0],
              [1, 2, 7, 4, 5, 8, 6, 3, 9, 100]], dtype=np.int)
multi_reduce_mean = tf.reduce_mean(data1)

truncated = tf.truncated_normal([3, 2],stddev=1)
with tf.Session() as sess: 
    r = sess.run(Y)
    r2 = sess.run(logY)
    r3 = sess.run(multi)
    r4 = sess.run(multi_reduce_mean)
    r5 = sess.run(truncated)
    
    data3 = np.array([[2,5,6],[7,6,1]])
    Y = np.array([[0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]])
    Y_ = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]])
    print(sess.run(tf.argmax(Y, 1))) # 返回某个维度（此处是1），最大值索引
    print(sess.run(tf.argmax(Y_, 1))) 
    equal = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    print(sess.run(equal))
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
    print(sess.run(accuracy))
    
    
    data_w = np.array([[1, 2, 7, 4, 5, 8, 6, 3, 9, 0],
              [1, 2, 7, 4, 5, 8, 6, 3, 9, 100],
              [1, 2, 7, 4, 5, 8, 6, 3, 9, 100],
              [1, 2, 7, 4, 5, 8, 6, 3, 9, 100]], dtype=np.int)
    print(data_w.shape)
    data_hsplit = np.hsplit(data_w, 10)
    print(data_hsplit)
    data_hsplit2 = np.array(data_hsplit);
    data_hsplit2 = np.squeeze(data_hsplit2)
    print(data_hsplit2.shape)
    