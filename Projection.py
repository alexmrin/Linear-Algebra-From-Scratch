import numpy as np

def projection(x, B):
    proj_matrix = np.dot(np.dot(B, np.linalg.inv(np.dot(np.transpose(B), B))), np.transpose(B))
    return np.dot(proj_matrix, x)

'''
B = np.array([2, -2, 5, 1, -1, 1]).reshape(3, 2)
print(B)
x = np.array([1, 2, 3]).reshape(3, 1)
print(projection(x, B))
'''