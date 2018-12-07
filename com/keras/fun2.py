from keras.models import Model, Sequential
from keras.layers import Input,Dense,Activation
from matplotlib import pyplot as plt

model = Sequential()
# [1000, 100] * [100, 32] = [1000, 32] 权值个数：32 * 100 
#  实际运行中根据batch_size 来的的话，总的权值比这个小得多(这句话是错的）,个数只与每层的输出维度有关系 
model.add(Dense(32, activation='relu', input_shape=(100,))) #input_dim=100
# [1000, 32] * [32, 1] = [1000, 1]  权值个数：32 * 1
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

import numpy as np
import keras as K

data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1)) #0,1,2,3,4...9
#plt.imshow(data)
print(labels)
one_hot_labels = K.utils.to_categorical(labels, num_classes=10)
print(one_hot_labels)
model.fit(data, one_hot_labels, batch_size = 64, epochs = 10)

