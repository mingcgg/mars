import tensorflow as tf 
import numpy as np
from numpy import float32

sess = tf.Session()

"""
    matrix truncted slice 分割
"""
random_num = tf.truncated_normal(shape=[4, 28, 28, 1], stddev=1)
print(random_num)
print(random_num.shape)
shift = tf.slice(random_num, [0, 0, 0, 0], [1, 28, 28, 1]) # begin size
print(shift.shape)
shift_sq= tf.squeeze(shift)
print(shift_sq.shape)
"""
    reduce_mean reduce_sum  按照维度求和求平均
"""
input_data = np.asarray([[ 0.7767287,   0.38119382], [-1.8936255,  -1.2566669 ]], float32)

#input_tensor = tf.truncated_normal([3], dtype=tf.float32) #不能直接用于求秒求平均，暂时还不知道为什么TOODO
reduce_mean = tf.reduce_mean(input_data)
reduce_sum = tf.reduce_sum(input_data)


"""
[[ 0.7767287,   0.38119382]
 [-1.8936255,  -1.2566669 ]]
-1.433727
-0.9385214
"""


x = np.linspace(-0.1, 0.1, 10)[:,np.newaxis]
w = np.reshape(np.linspace(-1, 1, 20), [1, 20])

y = tf.matmul(x, w)
