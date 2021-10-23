import numpy

a = numpy.array([
    [1,2],
    [2,3],
    [3,4]
])

print(a)
print("=======")

# 选择所有行的第0列
a[:,0] = 1

print(a)

# 选择第0行的所有列
a[0,:] = 9
print(a)