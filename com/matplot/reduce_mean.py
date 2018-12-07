import numpy as np
import tensorflow as tf
var1 = tf.log(1.0)
varE = tf.log(2.71828) # t.log是以E为底的对数函数
arr = np.array([[1,2,3,4,1],[5,6,7,8,9]],  dtype=float)
result = tf.reduce_mean(arr)
with tf.Session() as sess:
    print(sess.run(result))
    print(sess.run(var1))
    print(sess.run(varE))
    
