import numpy as np

def eigendecomp(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    P = eigenvectors
    D = np.diag(eigenvalues)
    P_inv = np.linalg.inv(P)
    return P, D, P_inv

'''
    A = np.array([1, 2, 3, 2, 4, 6, 3, 6, 7]).reshape(3,3)
    P, D, P_inv = eigendecomp(A)
    print(P)
    print(D)
    print(P_inv)
    print(np.dot(np.dot(P, D), np.linalg.inv(P)))
'''
