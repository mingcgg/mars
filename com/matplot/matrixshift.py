import numpy as np
from matplotlib import pyplot as plt 

X = np.random.random((5,5))


# (5, 5, 1, 4) - > (5, 5)
data = np.random.normal(size=(5, 5, 1, 4))
print(data)

temp = [];

for k, v in enumerate(data):
    for k2, v2 in enumerate(v):
        #print(k2)
        #print(v2)
        temp.append(v2[0][0]);
# print(temp)   
#r = tf.reshape(temp, 5, 5);   
# print(r)  
#plt.imshow(data)
#plt.show()

temp2 = [1,2,3,4,5,6,7,8]
temp2 = np.array(temp2);
print(temp2.shape)

temp2 = temp2.reshape(2, 4)
print(temp2)

#矩阵普通相乘
var1 = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=int)
var2 = np.array([[2,2,2],[2,2,2],[2,2,2]], dtype=int)
result = var1 * var2
print(result)
result = var1.dot(var2) # 矩阵相乘 shapes (2,3) and (2,3) not aligned: 3 (dim 1) != 2 (dim 0) M 与 N 相等!!!
print(result)
