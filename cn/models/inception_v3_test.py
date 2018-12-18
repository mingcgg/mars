import tensorflow as tf
import numpy as np

from cn.models import inception_v3 as inception
from tensorflow.python.framework import tensor_shape

slim = tf.contrib.slim

MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3

#from retrain_new
def add_input_distortions(flip_left_right, random_crop, random_scale, random_brightness):
    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(tensor_shape.scalar(), minval=1.0, maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, MODEL_INPUT_WIDTH)
    precrop_height = tf.multiply(scale_value, MODEL_INPUT_HEIGHT)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d, precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, axis=[0])
    cropped_image = tf.random_crop(precropped_image_3d, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_DEPTH])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(), minval=brightness_min, maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
    return jpeg_data, distort_result

class IncetionV3Test(tf.test.TestCase):
    
    def testSlim(self):
        inputs = np.random.uniform(-1, 0, [15, 301, 301, 3])
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
        with self.test_session() as sess:
            inputs = tf.placeholder(tf.float32, [batch_size, height, width, 3])
            net, end_points = inception.inception_v3(inputs, num_classes)
            tf.global_variables_initializer().run()
            pre_mixed = end_points['Mixed_7c']
            PreLogits = end_points['PreLogits']
            Logits = end_points['Logits']
            pre_mixed_out, PreLogits_out,Logits_out = sess.run([pre_mixed, PreLogits, Logits],  feed_dict={inputs: input_np})
            print(pre_mixed_out.shape)
            print(PreLogits_out.shape)
            print(Logits_out.shape)
        
if __name__ == '__main__':
    IncetionV3Test().testSlim()
    #tf.test.main()