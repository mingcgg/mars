import numpy as np

m2 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
print(m2)

def getLabel(data):
    temp = []
    for i in data:
        for idx in range(0,10):
            print(idx)
            j = i[idx]
            if j == 1:
                temp.append(idx)
    return temp

r = getLabel(m2)
print(r)