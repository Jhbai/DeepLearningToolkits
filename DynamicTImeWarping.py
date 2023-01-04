import numpy as np

def dist(v1, v2):
    return abs(v1 - v2)
def DTW(arr1, arr2):
    n1 = arr1.shape[0]
    n2 = arr2.shape[0]
    result = np.zeros(shape = (n1, n2))
    result[0, 0] = dist(arr1[0], arr2[0])
    for i in range(1, n1):
        result[i, 0] = dist(arr1[i], arr2[0]) + result[i - 1, 0]
    for i in range(1, n2):
        result[0, i] = dist(arr1[0], arr2[i]) + result[0, i - 1]
    for i in range(1, n1):
        for j in range(1, n2):
            result[i, j] = np.min([result[i - 1, j], result[i, j - 1], result[i - 1, j - 1]]) + dist(arr1[i], arr2[j])
    return result[n1 - 1, n2 - 1]