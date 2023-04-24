import numpy as np

def projection(x, B):
    proj_matrix = np.dot(np.dot(B, np.linalg.inv(np.dot(np.transpose(B), B))), np.transpose(B))
    return np.dot(proj_matrix, x)

B = np.array([1, 1]).reshape(2, 1)
x = np.array([-14, 4]).reshape(2, 1)
print(projection(x, B))