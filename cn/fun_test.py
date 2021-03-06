import tensorflow as tf 
import numpy as np
from numpy import float32
import random
from tensorflow.python.framework import tensor_shape

print(tensor_shape.scalar())
#sess = tf.Session()
input_data = [1, 2, 3, 4, 5, 6]
#print(input_data.shape)
input_data_reshape = np.reshape(input_data, newshape=[1, 6])
print(input_data_reshape.shape)
input_data_reshape = tf.reshape(input_data, [1, 6])
print(input_data_reshape)
"""
    matrix concat
"""
data1 = np.random.uniform(-1,1,[1,29,29,3])
data2 = np.random.uniform(-1,1,[1,29,29,3])
concat_result = tf.concat([data1, data2], 0)
print(concat_result.shape)

t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
t12 = tf.concat([t1, t2], 0)
t34 = np.concatenate((t1, t2), axis=0)
print(t12.shape)
print(t34)

data1 = np.zeros(10, dtype = np.float32)
data2 = np.zeros(10, dtype = np.float32)
data1[1] = 1.0
data1 = tf.expand_dims(data1,0)
data2 = tf.expand_dims(data2,0)
print(data1.shape)
concat_result = tf.concat([data1, data2], 0)
print(concat_result)

#从一个均匀分布[low,high)中随机采样
uniform = np.random.uniform(0,10,[2,10])
print(uniform)
print(random.randrange(10))
"""
    matrix truncted slice 分割
"""
random_num = tf.truncated_normal(shape=[4, 28, 28, 1], stddev=1)
print(random_num)
print(random_num.shape)
shift = tf.slice(random_num, [0, 0, 0, 0], [1, 28, 28, 1]) # begin size  , length
print(shift.shape)
shift_sq= tf.squeeze(shift)
print(shift_sq.shape)
"""
    reduce_mean reduce_sum  按照维度求和求平均
"""
input_data = np.asarray([[[ 1, 2, 3], [6, 7, 8]], [[ 2, 2, 3],[6, 7, 8]]], float32)

#input_tensor = tf.truncated_normal([3], dtype=tf.float32) #不能直接用于求秒求平均，暂时还不知道为什么TOODO
reduce_mean = tf.reduce_mean(input_data, [1, 2])
reduce_sum = tf.reduce_sum(input_data)
#print(sess.run(reduce_mean))

"""
[[ 0.7767287,   0.38119382]
 [-1.8936255,  -1.2566669 ]]
-1.433727
-0.9385214
"""


x = np.linspace(-0.1, 0.1, 10)[:,np.newaxis]
w = np.reshape(np.linspace(-1, 1, 20), [1, 20])

y = tf.matmul(x, w)
