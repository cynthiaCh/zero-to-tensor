import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

print("a:")
print(a)
print("a.shape:", a.shape)

print("sum axis=0:")
print(np.sum(a, axis=0)) # axis=0 means sum along the rows, so we get a 1D array with the sum of each column

print(np.sum(a, axis=0).shape)

print("sum axis=0 keepdims=True:")
print(np.sum(a, axis=0, keepdims=True))
print(np.sum(a, axis=0, keepdims=True).shape)

print("sum axis=1:")
print(np.sum(a, axis=1))
print(np.sum(a, axis=1).shape)

print("sum axis=1 keepdims=True:")
print(np.sum(a, axis=1, keepdims=True))
print(np.sum(a, axis=1, keepdims=True).shape)