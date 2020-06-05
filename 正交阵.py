import numpy as np

a = np.eye(3)
a1 = np.linalg.inv(a)
a2 = a.T
print(a)
print(a1)
print(a2)

