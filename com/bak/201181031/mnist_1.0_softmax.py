import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from matplotlib import pyplot as plt
from matplotlib import animation

mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
#属于分类问题 全连接网络：每一个像素点（特征）都与标准值连接28*28*10
# 训练方式是，取一批数据整体训练全部的权值
#label
#step 1 create model
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # [样本数量， 长，宽，通道数]
Y_ = tf.placeholder(tf.float32,[None, 10]) # [样本数量， 分类种数类别数] 存放实际值，用于与预测值对比

XX = tf.reshape(X, [-1, 784]) # 模型形状转换  行，列[样本数量， 28*28＝784列] 每一列的某一个表示一个像素点

#W = tf.Variable(tf.zeros([784, 10]))
# 尽量不要全0
W = tf.Variable(tf.zeros([784, 10]))#tf.ones([784, 10])   np.random.normal(size=(784, 10)).astype(np.float32)

b = tf.Variable(tf.zeros([10]))

#soft max 不会改变形状， 在最低维度上计算其总占比为100%的每个分量的占比 [[0.1, 0.1...0.7,0.001], [], []]
# 将所有像素点值与权值矩阵相乘，转换为概率输出,softmax:1、将总值为100%，2、突出数值相对较大的那个位置，这个之前写过代码进行验证的
Y = tf.nn.softmax(tf.matmul(XX, W) + b) #[－1, 748] * [784, 10] = [-1, 10] 

cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000 
# 这是整个程序最为关键的一句 log自然对数函数，x(0 - 1 此x(Y)处为softmax的计算结果0到1) -> y(0 到 负无穷) 每一张图片贡献自已单独的熵
# 与实际值Y_相乘， [0 0 0 1 0 0 ……]自己为1的位保留放大，最后得到一个整体的交叉熵 
# 20181026 明
# 1、tf.log(Y) 计算后的结果的形状依然是 [-1, 10] log为此模型的激函数
# 2、tf.log(Y) 计算后的结果 为0 到 负无穷[[－0.01, -0.2, -0.4, -0.3...-0.06], [], []...[]]
# 3、Y_ * tf.log(Y) [None, 10] * [-1 , 10] = [-1, 10] 相乘为普通相乘，不是矩阵相关（这里特别注意一下） ,形状不变(写段代码验证一下)
# 4、每次相乘的时候，10个值中只有一个为1，其它位置都是0，即为1的位置为相乘后会保留下来，从log函数曲线上可以看来，如果概率分布值越大得到的y值就越小（交叉熵就小，就越靠近正确答案）
#   如果此位置的概率分布值很小，就会得到一个负的很大的y值（交叉熵就大，就越偏离了正确答案），下次权值就会往交叉熵小的方向调整（按照设置的学习率调整）
# 5、cross_entropy是一个损失函数数值(tf.reduce_mean 结果是一个值）
allweights = tf.reshape(W, [-1])
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver() 

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

fig = plt.figure()

ax0 = None
ax1 = None
ax2 = None
ax3 = None
ax4 = None
ax5 = None
ax6 = None
ax7 = None
ax8 = None
ax9 = None
ax = None
img0 = None
img1 = None
img2 = None
img3 = None
img4 = None
img5 = None
img6 = None
img7 = None
img8 = None
img9 = None


def init_mpl():
    plot0 = fig.add_subplot(2, 5, 1)
    global img0
    global img1
    global img2
    global img3
    global img4
    global img5
    global img6
    global img7
    global img8
    global img9
    global ax0
    global ax1
    global ax2
    global ax3
    global ax4
    global ax5
    global ax6
    global ax7
    global ax8
    global ax9
    img0 = np.random.random(size = (28, 28))
    plot1 = fig.add_subplot(2, 5, 2)
    img1 = np.random.random(size = (28, 28))
    ax0 = plot0.imshow(img0, animated=True, cmap='jet', vmin=0.0, vmax=1.0, interpolation='nearest', aspect=1.0)
    ax1 = plot1.imshow(img1, animated=True, cmap='jet', vmin=0.0, vmax=1.0, interpolation='nearest', aspect=1.0)
    return ax0, ax1
    
def updateImg():
    global img0
    img0 = np.random.random(size = (28, 28))
    global img1
    img1 = np.random.random(size = (28, 28))

def training_step(i):
    global img0
    global img1
    #plt.pause(0.01)
    for k in range(20):
        batch_x,  batch_y = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={X:batch_x, Y_:batch_y}) 
        if i % 100 == 0:
            accuracy_out = sess.run(accuracy, feed_dict={X:batch_x, Y_:batch_y})
            print("cross entropy:", accuracy_out)
        WW = sess.run(W, feed_dict={X:batch_x, Y_:batch_y})
        #print(WW.shape)
        
        if k == 19:
            temp0 = [];
            temp1 = [];
            temp2 = [];
            temp3 = [];
            temp4 = [];
            temp5 = [];
            temp6 = [];
            temp7 = [];
            temp8 = [];
            temp9 = [];
            for k, v in enumerate(WW):
                temp0.append(v[0]);
                temp1.append(v[1]);
                temp2.append(v[2]);
                temp3.append(v[3]);
                temp4.append(v[4]);
                temp5.append(v[5]);
                temp6.append(v[6]);
                temp7.append(v[7]);
                temp8.append(v[8]);
                temp9.append(v[9]);
            array = np.array(temp0, dtype=float)
            img0 = array.reshape(28, 28)
            array1 = np.array(temp7, dtype=float)
            img1 = array1.reshape(28, 28)
def animate(i):
    training_step(i)
    ax0.set_data(img0)
    ax1.set_data(img1)
    return img0
        
ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=1000,
                              init_func=init_mpl,
                              interval=100,
                              blit=False)
        
plt.show()