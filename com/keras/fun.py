from keras.models import Model, Sequential
from keras.layers import Input,Dense,Activation

model = Sequential()
# [1000, 100] * [100, 32] = [1000, 32] 权值个数：32 * 100 
#  实际运行中根据batch_size 来的的话，总的权值比这个小得多(这句话是错的）,个数只与每层的输出维度有关系 
model.add(Dense(32, activation='relu', input_shape=(100,))) #input_dim=100
# [1000, 32] * [32, 1] = [1000, 1]  权值个数：32 * 1
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

import numpy as np

data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))
print(data)
model.fit(data, labels, batch_size = 64, epochs = 20)