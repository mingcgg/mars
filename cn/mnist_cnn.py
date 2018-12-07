import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
tf.set_random_seed(0)

mnist = mnist_data.read_data_sets("../data", one_hot=True, reshape=False, validation_size=0)

summaries_dir = './summaries'
if tf.gfile.Exists(summaries_dir):
    tf.gfile.DeleteRecursively(summaries_dir)
tf.gfile.MakeDirs(summaries_dir)


X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="x_input")
Y_ = tf.placeholder(tf.float32, shape=[None, 10], name="Y_label")

K = 4
L = 8
M = 16
N = 200

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
W3 = tf.Variable(tf.truncated_normal([5, 5, L, M], stddev=0.1))
W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))

B1 = tf.Variable(tf.zeros([K]) / 10)
B2 = tf.Variable(tf.zeros([L]) / 10)
B3 = tf.Variable(tf.zeros([M]) / 10)
B4 = tf.Variable(tf.zeros([N]) / 10)
B5 = tf.Variable(tf.zeros([10]) / 10)

stride = 1
# L1 output (-1, 28, 28, 4)
with tf.name_scope(name="Layer_1"):
    L1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding="SAME", name="conv_1") + B1)
stride = 2
# L2 output (28 - 2)/ 2 + 1 = 14 (-1, 14, 14, 8)
with tf.name_scope(name="Layer_2"):
    L2 = tf.nn.relu(tf.nn.conv2d(L1, W2, strides=[1, stride, stride, 1], padding="SAME", name="conv_2") + B2)

# L3 output (14 - 2)/ 2 + 1 = 7 (-1, 7, 7, 16)
with tf.name_scope(name="Layer_3"):
    L3 = tf.nn.relu(tf.nn.conv2d(L2, W3, strides=[1, stride, stride, 1], padding="SAME", name="conv_3") + B3)
    L3_reshape = tf.reshape(L3, [-1, 7 * 7 * M], name="L3_reshape")
# L4 output (-1, N) (-1, 200)
with tf.name_scope(name="Layer_4"):
    L4 = tf.nn.relu(tf.matmul(L3_reshape, W4, name='full_connect_1') + B4)

# L5 output  (-1, 10)
with tf.name_scope(name="Layer_5"):
    L5 = tf.matmul(L4, W5, name='full_connect_2') + B5

Y = tf.nn.softmax(L5, name="softmax")

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=L5, name="softmax_cross_entropy_with_logits")
loss = tf.reduce_mean(cross_entropy) * 100
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss) #学习率目前只能试

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
prediction = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.summary.FileWriter(summaries_dir, sess.graph)
for idx in range(2000):
    batch_x, batch_y = mnist.train.next_batch(100)
    _, LOSS, PREDICTION = sess.run([train_step,loss, prediction], feed_dict={X: batch_x, Y_: batch_y})
    if idx % 100 == 0:
        print(LOSS, PREDICTION)


batch_x, batch_y = mnist.train.next_batch(100)
LL1 = sess.run(L1, feed_dict={X: batch_x, Y_: batch_y})
shift = tf.squeeze(tf.slice(LL1, [13, 0, 0, 0], [1, 28, 28, 1])) # begin size
ss = sess.run(shift)
plt.imshow(ss, cmap = "inferno")
plt.colorbar() 
plt.show()