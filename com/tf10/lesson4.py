import tensorflow as tf
import numpy as np
#生成常量数据
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2
print(y_data)
#创建模型  y = f(x) = k * x + b
k = tf.Variable(0.)
b = tf.Variable(0.)
y = k * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.2)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for step in range(201):
        session.run(train)
        if(step % 20 == 0):
            print(step, session.run([loss,k, b]))
            

#标准差计算
a = np.array([0, 4, 7, 8, 9])
b = np.array([5, 6, 7, 8, 9])

print(np.std(a))
print(np.std(b))

