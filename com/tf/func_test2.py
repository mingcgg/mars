import tensorflow as tf
import numpy as np

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
    print(r)
    print(r2)
    print(r3)
    print(r4)
    print(r5)
