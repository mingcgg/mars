import tensorflow as tf
#fetch & feed

#Fetch 一个会话里面一次执行多个 任务opertion

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input1, input2)
mult = tf.multiply(input1, add)

with tf.Session() as session:
    result = session.run([mult, add])
    print(result)
    
#Feed
#创建占们符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as session:
    result = session.run(output, feed_dict={input1:[77.], input2:[2.]})
    print(result)
    