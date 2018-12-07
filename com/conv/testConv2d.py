import tensorflow as tf
#case 2
input = tf.Variable(tf.random_normal([10, 28, 28, 1]))
filter = tf.Variable(tf.random_normal([5, 5, 1, 4]))

op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')#VALID SAME

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    op2_out = sess.run(op2)
    print("case 2:", op2_out)
    print("case 2 shape:", op2_out.shape)