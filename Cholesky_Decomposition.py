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

    return l

def determinant(a):
    det = 1
    l = cholesky(a)
    for i in range(l.shape[0]):
        det *= l[i][i] ** 2
    return det

'''
    S = A^TA where A has linearly independent columns, is a symmetric positive definite matrix. We need this to be the case because 
if we multiply out the Cholesky Decomposition, we find that we must divide certain terms by the diagonal elements. Positive semidefinite
means that for any nonzero vector x, x^TA^TAx >= 0, which simplifies to (Ax)^T(Ax) >= 0. This is also the dot product of vector Ax
with itself. we know that this dot product can equal 0 whenever x is in the nullspace of A, which will give us (0)^T(0) = 0.
Choosing one of the standard basis vectors as x, we can get a diagonal term of S. If one of the standard basis vectors happen to be in
the nullspace of A, this results in the diagonal term of S becoming 0, which will cause an error due to division by 0.

    A reason why the Cholesky Decomposition is useful is it speeds up the calculation of determinants. Since S = LL^T, 
det(S) = det(L) * det(L^T). We know that for a triangular matrix, the determinant can be calculated by multiplying the diagonal terms
of the matrix. Since det(L) = det(L^T), we can say det(S) = det(L)^2. Therefore the determinant of S can be calculated by
multiplying the square of the diagonal terms on its triangular matrix.
'''

# columns linearly independent matrix A
A = np.array([4, 5, 2, 3, 5, 1, 19, 17, 3]).reshape(3, 3)

# constructing symmetric positive definite matrix
S = np.dot(np.transpose(A), A)
print(A)
print(S)

l = cholesky(S)

print("Your decomposed lower triangular matrix is")
print(l)

print("The determinant of your matrix is")
print(determinant(S))

