import tensorflow as tf


#创建图 启动图
m1 = tf.constant([[3, 3]])
m2 = tf.constant([[2], [3]])

product = tf.matmul(m1, m2)
print(product) #Tensor("MatMul:0", shape=(1, 1), dtype=int32)

session = tf.Session()

result = session.run(product)
print(result) #[[15]]

session.close()

with tf.Session() as session:
    result = session.run(product)
    print(result)
    
