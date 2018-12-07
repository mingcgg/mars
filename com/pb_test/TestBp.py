import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data as mnist_data
tf.set_random_seed(0)

mnist = mnist_data.read_data_sets("../../data", one_hot=True, reshape=False, validation_size=0)


def recognize(pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_file_path, mode='rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name='')
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            # 获取图中的节点，赋值进去运算，得出结果
            input_x = sess.graph.get_tensor_by_name("input:0")
            # print input_x
            out_softmax = sess.graph.get_tensor_by_name("softmax:0")
            # print out_softmax
            # out_label = sess.graph.get_tensor_by_name("output:0")
            # print out_label

            # img = io.imread(jpg_path)
            # img = transform.resize(img, (224, 224, 3))
            img_out_softmax = sess.run(out_softmax, feed_dict={input_x: np.reshape(mnist.test.images[2], [-1, 28, 28, 1])})

            print ("img_out_softmax:",img_out_softmax)
            prediction_labels = np.argmax(img_out_softmax, axis=1)
            print ("prediction label:",prediction_labels)
            print('true label:',mnist.test.labels[2])
recognize('C:/tmp/output_mine/pbfiles/mnist.pb')
    