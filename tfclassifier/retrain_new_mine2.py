import argparse
import hashlib
import os
import re
import sys
import random

from absl.flags import Flag
from tensorflow import gfile
from tensorflow.python.util import compat
from tensorflow.python.framework import graph_util
import numpy as np
import tensorflow as tf


FLAGS = None

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,  category):
    return get_image_path(image_lists, label_name, index, bottleneck_dir, category) + '.txt'
def get_image_path(image_lists, label_name, index, image_dir, category):
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.', label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    # full_path = 'D:/xxx/dataset' + 'a_ant' + ac212ac2.png
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path

def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})# core code
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

def get_or_create_bottleneck(sess, image_list, label_name, index, image_dir, category, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
    lable_list =  image_list[label_name]
    sub_dir = lable_list['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_list, label_name, index, bottleneck_dir, category)
    if not os.path.exists(bottleneck_path):
        print('creating bottleneck dir', bottleneck_dir)
        image_path = get_image_path(image_list, label_name, index, image_dir, category)
        if not os.path.exists(image_path):
            print('Image file is not exists %s', image_path)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as b_file:
            b_file.write(bottleneck_string)
    
    with open(bottleneck_path, 'r') as b_file:
        bottleneck_string = b_file.read()
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values;

def cache_bottlenecks(sess, image_list, image_dir, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
    
    ensure_dir_exists(bottleneck_dir)
    how_many = 0
    for label_name, label_lists in image_list.items():
        for category in ['training', 'testing', 'validation']:
            label_list = label_lists[category]
            for index, unused_base_name in enumerate(label_list):
                get_or_create_bottleneck(sess, image_list, label_name, index, image_dir, category, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)
                how_many += 1
                if how_many % 100 == 0:
                    print('bottleneck nums ' + str(how_many))
                    
def create_image_lists(image_dir, testing_percentage=10, validation_percentage=10):
    if not gfile.Exists(image_dir):
        print('Image diretory ' + image_dir + ' not exists')
        return None
    result = {}
    # gfile.Walk() ['当前路径名称str', '文件夹名称list', '文件名称list']
    #'D:/research/tensorflow/ai_challenger_zsl2018/DataSet/Animals2', ['A_ant', 'A_bear', 'A_butterfly'], [])
    #'D:/research/tensorflow/ai_challenger_zsl2018/DataSet/Animals2\\A_ant', [], 
    #'D:/research/tensorflow/ai_challenger_zsl2018/DataSet/Animals2\\A_bear', []
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    extensions = ['jpg', 'jpeg'] #, 'jpeg', 'JPG' same
    for index, sub_dir in enumerate(sub_dirs):
        if index == 0:
            continue
        dir_name = os.path.basename(sub_dir)
        file_list = []
        if dir_name == image_dir:
            continue
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        label_name = re.sub(r'[^a-z0-9]+', '_', dir_name.lower())
        
        training_images = []
        testing_images = []
        validation_images = []
        # TODO check
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                              (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                             (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result       
def create_inception_graph():
    """"  从已有的ＰＢ文件中创建图对象 Ｇraph
          Returns:
            bottleneck_tensor: shape=(1, 2048) pool_3/_reshape somftmax之前的最后一层结果
            jpeg_data_tensor: shape=(1, 299, 299, 3) DecodeJpeg/contents:0
            resized_input_tensor:
          """
    with tf.Session() as sess:
        modelfile_name = os.path.join(FLAGS.model_dir, 'classify_image_graph_def.pb')
        with gfile.FastGFile(modelfile_name, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            
            bottlenect_tensor, jpeg_data_tensor, resized_input_tensor = (tf.import_graph_def(graph_def, name='', return_elements=['pool_3/_reshape:0', 'DecodeJpeg/contents:0', 'ResizeBilinear:0']))
    return sess.graph, bottlenect_tensor, jpeg_data_tensor, resized_input_tensor

def add_final_opts(class_count, bottleneck_tensor):
    #key code
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(bottleneck_tensor, shape = [None, 2048], name='bottleneck_input_placeholder')
        truth_input  = tf.placeholder(tf.float32, shape = [None, class_count], name='truth_input')
    
    with tf.name_scope('weights'):
        weight_input = tf.Variable(tf.truncated_normal(shape=[2048, class_count], stddev=0.001), name='final_weight')
        
    with tf.name_scope('biases'):
        biase_input = tf.Variable(tf.zeros(shape=[class_count]), name='final_biase')
        
    with tf.name_scope('Wx_plus_b'):
        # (N, 2048) * (2048, class_count)
        logits = tf.matmul(bottleneck_input, weight_input) + biase_input
    
    final_tensor = tf.nn.softmax(logits, name='final_result')
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=truth_input)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
        
    with tf.name_scope('train'):   
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_mean)
    return (train_step, cross_entropy_mean, bottleneck_input, truth_input, final_tensor)

def add_evaluation_opt(input_tensor, truth_tensor):
    with tf.name_scope('accuracy'):
        with tf.name_scope('prediction'):
            prediction = tf.equal(tf.argmax(input_tensor, 1), tf.argmax(truth_tensor, 1), name='prediction')
        with tf.name_scope('prediction'):
            accuracy_step = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return accuracy_step;

def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  bottleneck_tensor):
   
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                          image_index, image_dir, category,
                                          bottleneck_dir, jpeg_data_tensor,
                                          bottleneck_tensor)
    ground_truth = np.zeros(class_count, dtype=np.float32)
    ground_truth[label_index] = 1.0
    bottlenecks.append(bottleneck)
    ground_truths.append(ground_truth)
    return bottlenecks, ground_truths
def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    
    graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (create_inception_graph())
    
    """
    dict: {'a butterfly': {'testing': ['0125fd33575ea877fdf9d04b261ddd4b.jpg', 
        '09ed634293d31d731d97c73aaf6ad75a.jpg', ...], 'training':[]...}, 'a ant':{}...}
    """
    image_list = create_image_lists(FLAGS.image_dir)
    print(image_list)
    sess = tf.Session()
    cache_bottlenecks(sess, image_list, FLAGS.image_dir, FLAGS.bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)
    
    train_step, cross_entropy_mean, bottleneck_input, truth_input, final_tensor = (add_final_opts(len(image_list), bottleneck_tensor))
    evaluation_step = add_evaluation_opt(final_tensor, truth_input)
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for idx in range(500):
        train_bottlenecks, train_ground_truth = (get_random_cached_bottlenecks(sess, image_list, 100, 'training',
                                  FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                                  bottleneck_tensor))
        sess.run(train_step, feed_dict = {bottleneck_input: train_bottlenecks, truth_input: train_ground_truth})
        if idx % 100 == 0:
            print('steps : %s', idx)
    
    
    test_bottlenecks, test_ground_truth = get_random_cached_bottlenecks(
        sess, image_list, 100, 'testing',
        FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
        bottleneck_tensor)
      
    test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks, truth_input: test_ground_truth})
    print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

    # Write out the trained graph and labels with the weights stored as constants.
    # 保存图和变量，及标签
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['final_result'])
    with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
        f.write('\n'.join(image_list.keys()) + '\n')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str, 
        default='D:/research/tensorflow/ai_challenger_zsl2018/DataSet/Animals2', 
        help='Path to Folders of labeled images '
    )
    parser.add_argument(
        '--summaries_dir',
        type=str, 
        default='./summaries', 
        help='Where to save summary logs for tensorboard'
    )
    parser.add_argument(
        '--model_dir',
        type=str, 
        default='C:/tengk/wangm/dev/workspace_py/tfClassifier/image_classification/inception', 
        help='Path to folder of inception v3 pb files'
    )
    parser.add_argument(
        '--bottleneck_dir',
        type=str, 
        default='./bottleneck_dir', 
        help='Path to folder of inception v3 pb files'
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='./tmp/output_graph.pb',
        help='Where to save the trained graph.'
        )
    parser.add_argument(
        '--output_labels',
        type=str,
        default='./tmp/output_labels.txt',
        help='Where to save the trained graph\'s labels.'
        )
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS, unparsed)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)