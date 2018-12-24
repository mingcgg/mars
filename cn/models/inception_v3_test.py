import tensorflow as tf
import numpy as np

from cn.models import inception_v3 as inception
from tensorflow.python.framework import tensor_shape

slim = tf.contrib.slim

class IncetionV3Test(tf.test.TestCase):
    
    def testSlim(self):
        inputs = tf.placeholder(tf.float32, [15, 301, 301, 3])
        #inputs = np.random.uniform(-1, 0, [15, 301, 301, 3])
        #(COLUMN - 2)/ 2 + 1 32-输出深度 [3,3]-卷积核大小  stride=2 差不多会减半
        net = slim.conv2d(inputs, 32, [3, 3], stride=2, scope='conv2d_1')
        # input net: [15, 151, 151, 32] out:[15, 1, 1, 32]  [151, 151]-为池化结果大小  max_pool2d
        net = slim.avg_pool2d(net, [151, 151], padding='VALID', scope='AvgPool_1a_avg_pool2d')
        print(net)
        #logits
        num_class = 10
        logits = slim.conv2d(net, num_class, [1, 1], scope='conv2d_logits')
        logits = tf.squeeze(logits, [1, 2]) #将第几维度为1的去掉  [15, 1, 1, 32]=>[15, 32]
        print(logits)
    
    def testGraphOutput(self):
        summaries_dir = './summaries'
        
        batch_size = 2
        height = 300
        width = 300
        num_classes = 10
        inputs = np.random.uniform(-1, 0, [batch_size, height, width, 3])
        net, end_points = inception.inception_v3(inputs, num_classes)
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter(summaries_dir, sess.graph)

    def testGraph(self):
        batch_size = 2
        height = 300
        width = 300
        num_classes = 10
        input_np = np.random.uniform(-1, 0, [batch_size, height, width, 3])
        with tf.Session() as sess:
            inputs = tf.placeholder(tf.float32, [batch_size, height, width, 3])
            net, end_points = inception.inception_v3(inputs, num_classes)
            init = tf.global_variables_initializer()
            sess.run(init)
            pre_mixed = end_points['Mixed_7c']
            PreLogits = end_points['PreLogits']
            Logits = end_points['Logits']
            pre_mixed_out, PreLogits_out,Logits_out = sess.run([pre_mixed, PreLogits, Logits],  feed_dict={inputs: input_np})
            print(pre_mixed_out.shape)
            print(PreLogits_out.shape)
            print(Logits_out.shape)
        
if __name__ == '__main__':
    IncetionV3Test().testGraph()
    #tf.test.main()