#學生:吳國良 學號: D20091105027

# 題 1.
import numpy as np
a = np.array([4, 5, 6])
print(a.dtype)
print(a.shape)
print(a[0])

# 題 2.
b = np.array( [[4, 5, 6],
               [1, 2, 3]] )
print(b.shape)
print(b[0,0],b[0,1],b[1,1])

#  題 3.
a = np.zeros( (3, 3),  dtype = int )
b = np.ones( (4, 5),  dtype = int )
c = np.identity( (4),  dtype = int )
d = np.random.randint( 1,  10,  dtype = int,  size = (3, 2) )
print(a)
print(b)
print(c)
print(d)

# 題 4.
a = np.array( [[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]] )
print(a)
print(a[2, 3], a[0, 0])

# 題 5.
b = a[0:2, 1:3]
print(b)

# 題 6.
c = a[1:3]
print(c)
print(c[-1][-1])

# 題 7.
a=np.array([[1, 2],
            [3, 4],
            [5, 6]])
print(a[[0, 1, 2], [0, 1, 0]])

# 題 8.
a = np.array([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]])
b = np.array([0, 2, 0, 1])
print(a[np.arange(4), b])

# 題 9.
a[np.arange(4),b] += 10;
print(a)

# 題 10.
x = np.array([1, 2])
print(x.dtype)

# 題 11.
x = np.array([1.0, 2.0])
print(x.dtype)

# 題 12.
x = np.array([[1, 2],
              [3, 4]], dtype = np.float64)
y = np.array([[5, 6],
              [7, 8]], dtype = np.float64)
print(x + y)
print(np.add(x, y))

# 題 13.
print(x - y)
print(np.subtract(x, y))

# 題 14.
print(x * y)
print(np.multiply(x, y))
print(np.dot(x, y))

# 題 15.
print(x / y)
print(np.divide(x, y))

# 題 16.
print( np.sqrt(x) )

# 題 17.
print( x.dot(y) )
print( np.dot(x,y) )

# 題 18.
print( np.sum(x) )
print( np.sum(x, axis =0))
print( np.sum(x, axis =1))

# 題 19.
print( np.mean(x) )
print( np.mean(x,axis = 0) )
print( np.mean(x,axis = 1) )

# 題 20.
x = x.T
print(x)

# 題 21.
print( np.exp(x) )

# 題 22.
print(np.argmax(x))
print(np.argmax(x, axis =0))
print(np.argmax(x,axis =1))

# 題 23.
import matplotlib.pyplot as plt
x = np.arange(0, 100, 0.1)
y = x * x
plt.plot(x, y)
plt.show()

# 題 24.
x = np.arange(0, 3*np.pi, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()
y = np.cos(x)
plt.plot(x, y)
plt.show()





