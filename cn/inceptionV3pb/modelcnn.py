import tensorflow as tf
import numpy as np
from cn.models import model_cnn as modelcnn
from tensorflow.python.framework import graph_util
from tensorflow import gfile
import os
import random
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
tf.set_random_seed(0)

mnist = mnist_data.read_data_sets("../../data", one_hot=True, reshape=False, validation_size=0)

from cn.data.ImagePreprocess import get_distortions

NUM_CLASSES = 8
BATCH_SIZE = 20
lr = 0.01
output_dir = './pbfile/inception_v3_train.pb'
IMAGE_DIR = 'D:/research/tensorflow/ai_challenger_zsl2018/DataSet/Animals'

def create_image_list(image_dir):
    if not gfile.Exists(image_dir):
        print('Image folder not exist', image_dir)
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    extension = 'jpg'
    for index, sub_dir in enumerate(sub_dirs):
        if index == 0:
            continue
        dir_name = os.path.basename(sub_dir) #A_ant A_bear A_butterfly...
        file_glob = os.path.join(sub_dir, '*.' + extension)
        file_list = []
        file_list.extend(gfile.Glob(file_glob))
        images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            images.append(base_name)
        result[dir_name] = {
            'dir': dir_name,
            'training': images
        }
    return result
def create_input_data_x(filename):
    if not gfile.Exists(filename):
        tf.logging.fatal('file is not exist', filename)
    image_data = gfile.FastGFile(filename, 'rb').read()
    decoded_image = get_distortions(image_data)
    return decoded_image
    
def get_random_images(image_list, howmany = 5):
    num_class = len(image_list.keys())
    input_x_list = []
    input_label_list = []
    keys = list(image_list.keys())
    for unuesed_idx in range(howmany):
        label_idx = random.randrange(num_class)
        label_name = keys[label_idx]
        class_list = image_list[label_name]['training']
        # random pick one image in label categroy
        class_len = len(class_list)
        random_n = random.randrange(class_len + 1)
        image_idx = random_n % class_len
        image_name = class_list[image_idx]
        full_path = os.path.join(IMAGE_DIR, label_name, image_name)
        
        input_x = create_input_data_x(full_path)
        input_x_list.append(input_x)
        #truth
        truth = np.zeros(num_class, dtype = np.float32)
        truth[label_idx] = 1.0
        truth = tf.expand_dims(truth, 0)
        input_label_list.append(truth)
    result_data = tf.concat(input_x_list, 0) 
    result_label = tf.concat(input_label_list, 0) 
    return result_data, result_label
def get_random_images2(image_list, howmany = 5):
    batch_x, batch_y = mnist.train.next_batch(20)
    return batch_x, batch_y

def create_cnn_model(inputs, num_classes):
    return modelcnn.cnn_base(inputs, num_classes)
    
    
def train():
    input_placeholder = tf.placeholder(tf.float32, [None, None, None, 3], name='input_placeholder_data')
    Y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES]) # reshape -1
    
    sess = tf.Session()
    
    # logits shape=[-1, NUM_CLASSES]
    logits = create_cnn_model(input_placeholder, NUM_CLASSES)
    # @see func_test.py 
    logits = logits / 10000
    Y = tf.nn.softmax(logits, name='final_result')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_) #-tf.log(Y) * Y_
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess.run(tf.global_variables_initializer())
    
    #input_data = np.random.uniform(0, 1, [BATCH_SIZE, 299, 299, 3])
    #y_label = np.random.uniform(0, 1, [BATCH_SIZE, NUM_CLASSES])
    image_list = create_image_list(IMAGE_DIR)
    for idx in range(50):
        input_data, y_truth = get_random_images(image_list, 10)
        with sess.as_default():
            input_data = input_data.eval()
            y_truth = y_truth.eval()
            input_data = input_data
        _, LOSS, ACCURACY,LOGITS,YY= sess.run([train_step, loss, accuracy, logits, Y], feed_dict={input_placeholder: input_data, Y_: y_truth})
        print(idx,LOSS, ACCURACY) #
    #tf.summary.FileWriter('./summaries', sess.graph)
    #save pb
    #output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['input_placeholder_data','final_result'])
    #with gfile.FastGFile(output_dir, 'wb') as f:
    #    f.write(output_graph_def.SerializeToString())
    
    sess.close()
def test():
    image_list = create_image_list(IMAGE_DIR)
    input_data, y_truth = get_random_images(image_list, 5)
    
if __name__ == '__main__':
    train()
    #create_image_list(IMAGE_DIR)