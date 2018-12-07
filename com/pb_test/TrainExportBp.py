import tensorflow as tf
import math
#mine 日期2018-11-08

from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from tensorflow.python.framework import graph_util

tf.set_random_seed(0)
mnist = mnist_data.read_data_sets("../../data", one_hot=True, reshape=False, validation_size=0)

summaries_dir = './summaries'
if tf.gfile.Exists(summaries_dir):
    tf.gfile.DeleteRecursively(summaries_dir)
tf.gfile.MakeDirs(summaries_dir)

def build_network(channel=1):
    lr = tf.placeholder(tf.float32)
    X = tf.placeholder(tf.float32, [None, 28, 28, channel], name='input')
    Y_ = tf.placeholder(tf.float32, [None, 10])
    K = 4 #first layer depth
    L = 8 #secode layer depth
    M = 16
    N = 200
    def weight_variable(shape, name='weights'):
        return tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1), name=name)
    def bias_variable(shape, name='biases'):
        return tf.Variable(tf.ones(shape=shape) / 10, name=name) #constant 0.1
    def conv2d(X, w, stride = 1):
        return  tf.nn.conv2d(X, w, [1, stride, stride, 1], padding='SAME')
    # TODO add pool dropout
    # conv1
    with tf.name_scope('conv_1') as scope:
        kernel = weight_variable([5, 5, 1, K])
        biases = bias_variable([K])
        conv = conv2d(X, kernel, 1)
        Y1 = tf.nn.relu(conv + biases, name=scope)
    # conv2
    with tf.name_scope('conv_2') as scope:
        kernel = weight_variable([5, 5, K, L])
        biases = bias_variable([L])
        conv = conv2d(Y1, kernel, 2) # stride = 2
        Y2 = tf.nn.relu(conv + biases, name=scope)
    # conv3
    with tf.name_scope('conv_3') as scope:
        kernel = weight_variable([4, 4, L, M])
        biases = bias_variable([M])
        conv = conv2d(Y2, kernel, 2) # stride = 2
        Y3 = tf.nn.relu(conv + biases, name=scope)
    # full connect layer 4
    with tf.name_scope('full_connect_4') as scope:
        w = weight_variable([7 * 7 * M, N])
        biases = bias_variable([N])
        Y3_reshape = tf.reshape(Y3, [-1, 7 * 7 * M])
        Y4 = tf.nn.relu(tf.matmul(Y3_reshape, w) + biases, name=scope)
    # full connect layer 5
    with tf.name_scope('full_connect_5') as scope:
        w = weight_variable([N, 10])
        biases = bias_variable([10])
        logits = tf.matmul(Y4, w) + biases
        
    Y = tf.nn.softmax(logits, name='softmax')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=Y_)) * 100
    tf.summary.scalar('cross_entropy', cross_entropy)
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)      
    prediction_label = tf.argmax(Y, axis=1, name='final_output') 
    #accuracy
    equal = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    # equal {x: X, y_: Y_ }
    return dict(
        x = X,
        y_ = Y_,
        optimizer = train_step,
        accuracy = accuracy,
        cost = cross_entropy,
        learningRate = lr
    )
def train_network(model, batch_size, epochs, pb_file_path):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
        for epoch_index in range(epochs):
            for i in range(1000):
                max_learning_rate = 0.003
                min_learning_rate = 0.0001
                decay_speed = 2000.0
                learning_rate = 0.001 #min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
                
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                #{X: batch_X, Y_: batch_Y, lr: learning_rate}
                feed = {
                    model['x']:  batch_x,
                    model['y_']:  batch_y,
                    model['learningRate']: learning_rate
                }
                sess.run(model['optimizer'], feed_dict = feed)
                if i % 100 == 0:
                    print('step', i, 'acc', sess.run(model['accuracy'], feed), 'loss', sess.run(model['cost'],feed), ' lr:', learning_rate)
                train_summary, _ = sess.run([merged, model['optimizer']], feed_dict = feed)
                train_writer.add_summary(train_summary, i)
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['final_output'] )
        with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
    
def main():
    model = build_network()
    train_network(model, 100, 2, 'C:/tmp/output_mine/pbfiles/mnist.pb')
    
main()    

"""
TODO 
1 pool dropout
2 tensorboard view
如何只导出一部分PB,graph
"""