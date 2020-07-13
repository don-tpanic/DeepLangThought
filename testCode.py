import numpy as np

a = np.random.random((16, 1000))
b = np.random.random((1000, 768))

for _ in range(10000):
    dot = np.dot(a, b)
print('ok')