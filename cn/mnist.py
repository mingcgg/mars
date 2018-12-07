import cn
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
# 数据
mnist = mnist_data.read_data_sets("../data", one_hot=True, reshape=False, validation_size=0)

#batch_y is labels shape=(10, 10)
(batch_x, batch_y) = mnist.train.next_batch(10)

#(10, 28, 28, 1)
print(batch_x.shape)

input_x = batch_x[0];
label_y = batch_y[0];
#28, 28, 1)
print(input_x.shape)
print(label_y)
with cn.tf.Session() as sess:
    print(sess.run(cn.tf.argmax(label_y, 0))) # argmax 后面一个参数表示维度

#draw_x = cn.np.reshape(input_x, [28, 28])
draw_x = cn.np.squeeze(input_x)
#(28, 28)
print(draw_x.shape)

#(28, 28)
one_cube = cn.np.reshape(draw_x, (784,))
#1.0
#print(max(one_cube))
#print(cn.np.mean(one_cube))

plt.imshow(draw_x)
plt.colorbar()
plt.show()
