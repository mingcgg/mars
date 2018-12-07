import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train[0].shape)
print(x_test.shape[0], 'test samples')

p1 = x_train[1].reshape(28, 28)
#p1 = p1 * 100;
print(p1)

plt.imshow(p1, cmap = "inferno")
plt.colorbar() 
plt.show()

