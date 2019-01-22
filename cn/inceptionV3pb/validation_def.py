import tensorflow as tf
import numpy as np
from tensorflow import gfile
import os

model_v3 = './pbfile/classify_image_graph_def.pb'

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def create_inception_graph():
    """"  从已有的ＰＢ文件中创建图对象 Ｇraph
    20181219
    graph node l加：0 才是tensor不加的话是操作符
    TODO constant retrain中的东西 暂时没有跑通
          Returns:
            bottleneck_tensor: shape=(1, 2048) pool_3/_reshape somftmax之前的最后一层结果
            jpeg_data_tensor: shape=(1, 299, 299, 3) DecodeJpeg/contents:0
            resized_input_tensor:
          """
    modelfile_name = model_v3
    with gfile.FastGFile(modelfile_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
        jpage_data_tensor, final_result_tensor = (tf.import_graph_def(graph_def, name='', return_elements=['DecodeJpeg/contents:0', 'pool_3/_reshape:0']))
    return jpage_data_tensor, final_result_tensor

jpage_data_tensor, final_tensor = create_inception_graph()
file_list = []
image_path = '../data/ant_1.jpg'  #1173
image_path = 'D:\\research\\tensorflow\\ai_challenger_zsl2018\\DataSet\\Animals\\A_bear'
file_glob = os.path.join(image_path, '*.jpg')
file_list.extend(gfile.Glob(file_glob))

softmax_tensor= sess.graph.get_tensor_by_name('softmax:0')
print(softmax_tensor.shape)
print(file_list)
for file_name in file_list:
    image_data = gfile.FastGFile(file_name, 'rb').read()
    result = sess.run(final_tensor, feed_dict={jpage_data_tensor: image_data})
    r = result[0]
    #print(r)
    #print(max(r))
    tt = tf.argmax(result, 1)
    print(sess.run(tt))


sess.close()