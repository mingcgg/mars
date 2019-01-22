import tensorflow as tf
from PIL import Image
import scipy

FILE_PATH = 'C:/tmp/cifar10_data/cifar-10-batches-bin/data_batch_1.bin'

def load_data(filename):
    record_len = 32*32*3 + 1
    reader = tf.FixedLengthRecordReader(record_bytes = record_len)
    key, value = reader.read(filename)
    
    record_bytes = tf.decode_raw(value, tf.uint8)

    label = tf.cast(tf.strided_slice(record_bytes, [0], [1]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [1], [record_len]), [3, 32, 32])
    # Convert from [depth, height, width] to [height, width, depth].
    uint8image = tf.transpose(depth_major, [1, 2, 0])

    return uint8image, label, depth_major

def train():
    arr = [FILE_PATH]
    filename_queue = tf.train.string_input_producer(arr) # good key code
    with tf.Session() as sess: 
        init = tf.local_variables_initializer() #global_variables_initializer()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess) # key code
        uint8image, label, image_data = load_data(filename_queue)
        print(sess.run(uint8image))
# from mine.py
def saveImg(sess, imageData, path):
    depth = sess.run(imageData)
    R = Image.fromarray(depth[:1, :, :].reshape(32, 32))
    G = Image.fromarray(depth[1:2, :, :].reshape(32, 32))
    B = Image.fromarray(depth[2:3, :, :].reshape(32, 32))
    img = Image.merge("RGB", (R, G, B))
    img.save(path)
    
def test():    
    arr = [FILE_PATH]
    filename_queue = tf.train.string_input_producer(arr, shuffle=False, num_epochs=3) 
    #reader = tf.WholeFileReader()
    #key, value = reader.read(filename_queue)
    uint8image, label, img_data = load_data(filename_queue)
    with tf.Session() as sess:
        init = tf.local_variables_initializer()
        sess.run(init)
        
        tf.train.start_queue_runners(sess=sess)
        meta = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        for i in range(20):
            data, data_key = sess.run([img_data, label])
            data_key = data_key[0]
            print(data_key)
            #saveImg(sess, img_data, '../data/temp/out_%d.jpg' % i)
            scipy.misc.toimage(data).save('../data/temp/%d_%s.jpg' % (i, meta[data_key]))
            #with open('../data/temp/out_%d.jpg' % i, 'wb') as fd:
            #   fd.write(data)
    
if __name__ == '__main__':
    train()
