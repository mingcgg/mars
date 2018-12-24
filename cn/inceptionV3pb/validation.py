import tensorflow as tf
import numpy as np
from tensorflow import gfile

model_v3 = './pbfile/inception_v3_train.pb'

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
        
        input_tensor, final_result_tensor = (tf.import_graph_def(graph_def, name='', return_elements=['input_placeholder_data:0', 'final_result:0']))
    return sess.graph, input_tensor, final_result_tensor

graph, input_tensor, final_tensor = create_inception_graph()
print(input_tensor)
print(final_tensor)

input_x = sess.graph.get_tensor_by_name('input_placeholder_data:0')
input_y = sess.graph.get_tensor_by_name('final_result:0')
print(input_x)
print(input_y)

#method 1
input_data = np.random.uniform(0, 1, [1, 299, 299, 3])
final_result = sess.run(input_y, feed_dict={input_x: input_data})
print(final_result)

#method 2
final_result = sess.run(final_tensor, feed_dict={input_tensor: input_data})
print(final_result)

sess.close()