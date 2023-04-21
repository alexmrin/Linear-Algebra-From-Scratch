# Math-for-ML
### Implementing math concepts taught in this book into code and summarizing important concepts and techniques

**The [textbook](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) used to study**

# ______________________________________



### Cholesky Decomposition

*The Cholesky Decomposition decomposes a symmetric positive definite matrix into a lower triangular matrix and its transpose*

Let $S = A^TA$ where $A$ has linearly independent columns, be a symmetric positive definite matrix. We need this to be the case because 
if we multiply out the Cholesky Decomposition, we find that we must divide certain terms by the diagonal elements. Positive semidefinite
means that for any nonzero vector $x$, $x^TA^TAx \geq 0$, which simplifies to $(Ax)^T(Ax) \geq 0$. This is also the dot product of vector $Ax$
with itself. we know that this dot product can equal $0$ whenever $x$ is in the nullspace of $A$, which will give us $(0)^T(0) = 0$.
Choosing one of the standard basis vectors as x, we can get a diagonal term of $S$. If one of the standard basis vectors happen to be in
the nullspace of $A$, this results in the diagonal term of $S$ becoming $0$, which will cause an error due to division by $0$.

A reason why the Cholesky Decomposition is useful is it speeds up the calculation of determinants. Since $S = LL^T$, 
$det(S) = det(L) * det(L^T)$. We know that for a triangular matrix, the determinant can be calculated by multiplying the diagonal terms
of the matrix. Since $det(L) = det(L^T)$, we can say $det(S) = det(L)^2$. Therefore the determinant of $S$ can be calculated by
multiplying the square of the diagonal terms on its triangular matrix.
