import numpy as np

arr1 = np.arange(8).reshape((2, 2, 2))
print(arr1)

arr2=arr1.transpose(1,2)
print(arr2)
