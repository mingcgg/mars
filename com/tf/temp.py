import matplotlib.pyplot as plt
import com.tf.input_data as input_data
import numpy as np

import random as ran

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


def TRAIN_SIZE(num):
    print("Total Training Images in Dataset = " + str(mnist.train.images.shape))
    x_train = mnist.train.images[:num,:]
    print ("x_train Examples Loaded = " + str(x_train.shape))
    y_train = mnist.train.labels[:num,:]
    print ("y_train Examples Loaded = " + str(y_train.shape))
   
    print(y_train[num])
    label = y_train[num].argmax(axis=0)
    
    image = x_train[num].reshape([28,28])
    
    plt.title("Example: %d  Label: %d" % (num, label))
    
    plt.imshow(image, cmap=plt.get_cmap("gray_r"))
    
    plt.show()
    return x_train, y_train

x_train, y_train = TRAIN_SIZE(55000)


#plt.plot([1,3,5,7],[12,5,8,11])
#plt.show()