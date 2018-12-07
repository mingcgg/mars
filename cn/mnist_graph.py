import tensorflow as tf

summaries_dir = './summaries'
if tf.gfile.Exists(summaries_dir):
    tf.gfile.DeleteRecursively(summaries_dir)
tf.gfile.MakeDirs(summaries_dir)


#full conncet network
Y_ = tf.placeholder(tf.float32, shape=[None, 10], name='Y_input_labels')
X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x_data_input')

# 每一列是一个特征
input_x = tf.reshape(X, shape=[-1, 784], name='reshape')
# train variables
weights = tf.Variable(tf.truncated_normal(shape=[784, 10]), name='weights')
biases = tf.Variable(tf.truncated_normal(shape=[10]), name='biases')

# shape = (-1, 10)
logits = tf.matmul(input_x, weights, name='matmul') + biases
# softmax 回归 ： 1、根据输入的种类，将数值转化为比例分布， 2、总值为1，分量都是0点几 0。8，~ 0。01， 3、为了进一步突出数值相对较大的位置，4、 不会改变形状
Y = tf.nn.softmax(logits, name='softmax')
#参见 mnist_1.0_softmax.py中的注释
with tf.name_scope('cross_entropy'):
    cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y) + Y * tf.log(Y_), name='reduce_mean') * 1000
# TODO learning rate
tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #tensorboard --logdir=.
    tf.summary.FileWriter(summaries_dir, sess.graph)



