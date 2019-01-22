import tensorflow as tf

"""
    inputs: [batch_size, 299, 299, 3]
    
"""
def cnn_base(inputs, num_classes):
    with tf.variable_scope('cnn_base'):
        stddev =  0.1
        W1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=stddev))
        W2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=stddev))
        W3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=stddev))
        W4 = tf.Variable(tf.truncated_normal([75 * 75 * 128, 200], stddev=stddev))
        W5 = tf.Variable(tf.truncated_normal([200, num_classes], stddev=stddev))
        
        B1 = tf.Variable(tf.zeros([32]) / 10)
        B2 = tf.Variable(tf.zeros([64]) / 10)
        B3 = tf.Variable(tf.zeros([128]) / 10)
        B4 = tf.Variable(tf.zeros([200]) / 10)
        B5 = tf.Variable(tf.zeros([num_classes]) / 10)
        # L1 output (-1, 299, 299, 32)
        with tf.name_scope(name="Layer_1"):
            L1 = tf.nn.relu(tf.nn.conv2d(inputs, W1, strides=[1, 1, 1, 1], padding="SAME", name="conv_1") + B1)
        # L2 output (299 - 2)/ 2 + 1 = 150  (-1, 150, 150, 64)
        with tf.name_scope(name="Layer_2"):
            L2 = tf.nn.relu(tf.nn.conv2d(L1, W2, strides=[1, 2, 2, 1], padding="SAME", name="conv_2") + B2)
        # L3 output (150 - 2)/ 2 + 1 = 75 (-1, 75, 75, 128)
        with tf.name_scope(name="Layer_3"):
            L3 = tf.nn.relu(tf.nn.conv2d(L2, W3, strides=[1, 2, 2, 1], padding="SAME", name="conv_3") + B3)
            L3_reshape = tf.reshape(L3, [-1, 75 * 75 * 128], name="L3_reshape")
        # L4 output (-1, N) (-1, 200)
        with tf.name_scope(name="Layer_4"):
            L4 = tf.nn.relu(tf.matmul(L3_reshape, W4, name='full_connect_1') + B4)
        # L5 output  (-1, 10)
        with tf.name_scope(name="Layer_5"):
            logits = tf.matmul(L4, W5, name='full_connect_2') + B5
    return logits

def cnn_base_v2(inputs, num_classes):
    with tf.variable_scope('cnn_base'):
        stddev = 0.1
        W1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=stddev))
        W2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=stddev))
        W3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=stddev))
        W4 = tf.Variable(tf.truncated_normal([75 * 75 * 128, 200], stddev=stddev))
        W5 = tf.Variable(tf.truncated_normal([200, num_classes], stddev=stddev))
        
        B1 = tf.Variable(tf.zeros([32]) / 10)
        B2 = tf.Variable(tf.zeros([64]) / 10)
        B3 = tf.Variable(tf.zeros([128]) / 10)
        B4 = tf.Variable(tf.zeros([200]) / 10)
        B5 = tf.Variable(tf.zeros([num_classes]) / 10)
        # L1 output (-1, 299, 299, 32)
        with tf.name_scope(name="Layer_1"):
            L1 = tf.nn.conv2d(inputs, W1, strides=[1, 1, 1, 1], padding="SAME", name="conv_1") + B1
        # L2 output (299 - 2)/ 2 + 1 = 150  (-1, 150, 150, 64)
        with tf.name_scope(name="Layer_2"):
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 2, 2, 1], padding="SAME", name="conv_2") + B2
        # L3 output (150 - 2)/ 2 + 1 = 75 (-1, 75, 75, 128)
        with tf.name_scope(name="Layer_3"):
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 2, 2, 1], padding="SAME", name="conv_3") + B3
            L3_reshape = tf.reshape(L3, [-1, 75 * 75 * 128], name="L3_reshape")
        # L4 output (-1, N) (-1, 200)
        with tf.name_scope(name="Layer_4"):
            L4 = tf.matmul(L3_reshape, W4, name='full_connect_1') + B4
        # L5 output  (-1, 10)
        with tf.name_scope(name="Layer_5"):
            logits = tf.matmul(L4, W5, name='full_connect_2') + B5
    return logits