import tensorflow as tf
import numpy as np

# tf.argmax tf.equal

# pic pix number
m3 = np.array([-1.0860619e+04, -3.1066400e+04,  8.4361855e+03,  9.1545703e+03, 6.3926133e+03,  2.5773247e+03, -1.1056079e+03,  9.8716855e+03])
m3 = m3 / 10000
#softmax 这个函数，传入的参数只有'大小合适'的 时候才会有比较正确的结果 ,好像是参数值小一点比较好
Y_m3 = tf.nn.softmax(m3)
m2 = np.array([[0, 0, 0, 0, 0.14917259, 0.210407, 0.669037, 0.8345183, 0, 0],
           [1, 2, 7, 4, 5, 8, 6, 3, 9, 0],
           [-10, 2, 7, 4, 5, 8, 6, 3, 9, 0]], dtype=np.float32)
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
    print(sess.run(Y_m3))
    print(r)
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
    