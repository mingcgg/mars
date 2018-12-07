import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

mnist = mnist_data.read_data_sets("../data", one_hot=True, reshape=False, validation_size=0)

summaries_dir = './summaries'
# if tf.gfile.Exists(summaries_dir):
#     tf.gfile.DeleteRecursively(summaries_dir)
# tf.gfile.MakeDirs(summaries_dir)

X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='X_input_data')
Y_ = tf.placeholder(tf.float32, shape=[None, 10], name='Y_label_input_data')
X_reshape = tf.reshape(X, [-1, 784])

W1 = tf.Variable(tf.truncated_normal(shape=[784, 200]))
W2 = tf.Variable(tf.truncated_normal(shape=[200, 100]))
W3 = tf.Variable(tf.truncated_normal(shape=[100, 50]))
W4 = tf.Variable(tf.truncated_normal(shape=[50, 20]))
W5 = tf.Variable(tf.truncated_normal(shape=[20, 10]))

B1 = tf.zeros([200])
B2 = tf.zeros([100])
B3 = tf.zeros([50])
B4 = tf.zeros([20])
B5 = tf.zeros([10])
# sigmoid 1 / (1 + e^-x) https://zh.numberempire.com/graphingcalculator.php y[0~1] x 在+-5时y趋近于0
# 激活函数，activation 
Y1 = tf.sigmoid(tf.matmul(X_reshape, W1) + B1, name='Y1') #(? * 200)  sigmoid : y = 1 / (1 + exp(-x))
Y2 = tf.sigmoid(tf.matmul(Y1, W2) + B2, name='Y2')
Y3 = tf.sigmoid(tf.matmul(Y2, W3) + B3, name='Y3')
Y4 = tf.sigmoid(tf.matmul(Y3, W4) + B4, name='Y4')

logits = tf.matmul(Y4, W5) + B5 #(? * 10)
Y = tf.nn.softmax(logits, name='softmax')

# Y_与 Y 的形状相同 shape=[-1, 10]
# Y(0~1) Y(0或者1) log(在[0-1]中取负无穷到0)
loss = - tf.reduce_mean(Y_ * tf.log(Y), name='cross_entropy') * 1000 # * 1000 reduce_mean

# sum lrate=0.04: current accuracy: 4900 0.97 7.691499
# mean lrate=0.005: current accuracy: 4900 0.96 19.500647
# 两种方式好像差不多
# https://www.jianshu.com/p/d99b83f4c1a6
train_step = tf.train.AdamOptimizer(0.005).minimize(loss) #

correct_prediction = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
init =tf.global_variables_initializer()
sess.run(init)

for idx in range(5000):
    batch_x, batch_y = mnist.train.next_batch(100)
    _, accuracy_out, loss_out = sess.run([train_step, accuracy, loss], feed_dict={X:batch_x, Y_:batch_y})
    if idx % 100 == 0:
        print('current accuracy:',idx, accuracy_out,loss_out)

# 权值共享，即每一次取一批样本进行训练的时候，都会利用之前已经训练好的权值进行正确性评估
#tf.summary.FileWriter(summaries_dir, sess.graph)

