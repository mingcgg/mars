from pylab import *

def f(x,y): return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 - y**2)

n = 5
x = np.linspace(-3, 3, 4*n)
y = np.linspace(-3, 3, 3*n)
X,Y = np.meshgrid(x, y)
print(x)
print(y)
print(X)
print(Y)
r = f(X,Y)
print(r.shape)
imshow(r), show()