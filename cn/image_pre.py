import tensorflow as tf
from tensorflow.python.platform import gfile
from cn.data import ImagePreprocess as imagePre
from pylab import *

image_placeholder, distort_result = imagePre.add_input_distortions(True, 0, 0, 100)

init = tf.global_variables_initializer()

def process1():
    with tf.Session() as sess:
        sess.run(init)
        #input_data = tf.random_normal([2, 299, 299, 3])
        input_data = gfile.FastGFile('./data/ant_1.jpg', 'rb').read()
        result = sess.run(distort_result, feed_dict={image_placeholder: input_data})
        print(result.shape)
        img = tf.slice(result, [0,0,0,0], [1,299,299,1])
        img = tf.squeeze(img)
        img = img.eval(session=sess)
        print(img.shape)
        imshow(img, cmap = "inferno")
        show()
        
def process_write():
    jpeg_data = gfile.FastGFile('./data/ant_1.jpg', 'rb').read()
    #temp = tf.image.decode_jpeg(jpeg_data)
    distort_result = imagePre.get_distortions(jpeg_data)
    print(distort_result)
    with tf.Session() as sess:
        sess.run(init)
        result = sess.run(distort_result)
        print(result.shape)
        
if __name__ == '__main__':
    process_write()


