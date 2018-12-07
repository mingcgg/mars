import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1]))

y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 1])

#cost = tf.reduce_sum(tf.pow(y_ - y), 2)
cost = tf.reduce_mean(tf.square(y_-y))

train_step = tf.train.GradientDescentOptimizer(0.0000001).minimize(cost)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(1000):
    
    xs = np.array([[i]])
    ys = np.array([[i * 2]])
    
    feed = {x : xs, y_ : ys}
    
    sess.run(train_step, feed_dict=feed)
    print("After %d iteration:" % i)
    print("W: %f" % sess.run(W))
    print("b: %f" % sess.run(b))
    print("cost: %f" % sess.run(cost, feed_dict=feed))
    
    # session.run(train_step, feed_dict=feed)
    #print(session.run(W))
    # session.run(b)
    # session.run(cost, feed_dict=feed)
    
    