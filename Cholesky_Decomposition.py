import numpy as np

def cholesky(a):
    dim = a.shape[0]
    l = np.zeros(a.shape)
    for j in range(0, dim):
        for i in range (j, dim):
            if (i == j):
                temp_l = a[i][j]
                for count in range(i):
                    temp_l -= l[i][count] ** 2
                l[i][j] = np.sqrt(temp_l)
            else:
                temp_l = a[i][j]
                for count in range(i):
                    temp_l -= l[i][count] * l[j][count]
                l[i][j] = temp_l / l[j][j]

    return l, np.transpose(l)

def determinant(a):
    det = 1
    l = cholesky(a)
    for i in range(l.shape[0]):
        det *= l[i][i] ** 2
    return det

'''
# columns linearly independent matrix A
A = np.array([4, 5, 2, 3, 5, 1, 19, 17, 3]).reshape(3, 3)

# constructing symmetric positive definite matrix
S = np.dot(np.transpose(A), A)
print(A)
print(S)

l, l_t = cholesky(S)

print("Your decomposed lower triangular matrix is")
print(l)

print("The determinant of your matrix is")
print(determinant(S))
'''
