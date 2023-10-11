# Linear-Algebra-From-Scratch
### Implementing math concepts taught in this book into code and summarizing important concepts and techniques

**The [textbook](https://course.ccs.neu.edu/ds4420sp20/readings/mml-book.pdf) used to study**
## Table of Contents
- [Projections](#projections)
- [Gram-Schmidt Orthogonalization](#gram-schmidt-orthogonalization)
- [Cholesky Decomposition](#cholesky-decomposition)
- [Diagonalization](#diagonalization)
- [Eigendecomposition](#eigendecomposition)
- [Singular Value Decomposition](#singular-value-decomposition)

# ______________________________________



### Projections

*Projections are a unique type of linear mapping where applying a projection to a vector twice is equal to applying it once. Projections are useful for dimensionality reduction and simplifying data.*

Let $V$ be a vector space and $U \subseteq V$ be a subspace of $V$. A linear mapping $\pi : V \rightarrow U$ is a projection if $\pi^{2} = \pi$.

Let us look at how the math behind projections work. we will assume that $(\mathbf{b_1}, \dots, \mathbf{b_m})$ is an ordered basis of $U$. The projection $\pi_U(\textbf{x})$ is the projection of $\textbf{x}$ onto $U$. We want to minimize the distance between $\textbf{x}$ and $\pi_U(\textbf{x})$, meaning that the vector $(\textbf{x} -  \pi_U(\textbf{x}))$ will be orthogonal to every basis vector in $U$. Using the dot product as the inner product, we get $$\langle \mathbf{x} -  \pi_U(\mathbf{x}), \mathbf{b_1} \rangle = \mathbf{b_1}^T(\mathbf{x} -  \pi_U(\mathbf{x})) = \mathbf{0}$$ $$\vdots$$ $$\langle \mathbf{x} -  \pi_U(\mathbf{x}), \mathbf{b_m} \rangle = \mathbf{b_m}^T(\mathbf{x} -  \pi_U(\mathbf{x})) = \mathbf{0}$$ Since we know $\pi_U(\mathbf{x}) \in U$, we can rewrite $\pi_U(\mathbf{x})$ as a linear combination of the basis vectors of $U$. $$\pi_U(\mathbf{x}) = \lambda_1\mathbf{b_1} + \dots + \lambda_m\mathbf{b_m} = \sum_{i=1}^m \lambda_i\mathbf{b_i}$$ Letting $\mathbf{\lambda} = [\lambda_1, \dots, \lambda_m]^T$ and $\mathbf{B} = [\mathbf{b_1}, \dots, \mathbf{b_m}]$, we get $\pi_U(\mathbf{x}) = \mathbf{B\lambda}$. Substituting, we now have $$\mathbf{b_1}^T(\mathbf{x} -  \mathbf{B\lambda}) = \mathbf{0}$$ $$\vdots$$ $$\mathbf{b_m}^T(\mathbf{x} -  \mathbf{B\lambda}) = \mathbf{0}$$ This is simply a simultaneous homogenous equation that can be written as: $$\mathbf{B}^T(\mathbf{x} -  \mathbf{B\lambda}) = \mathbf{0}$$ Note that $\mathbf{x}$ and $\mathbf{B\lambda}$ are both vectors so by the linear property of matrices, we expand into $$\mathbf{B}^T\mathbf{x} = \mathbf{B}^T\mathbf{B\lambda}$$ We are solving for the coordinates with respect to the subspace $U$, $\lambda$. Since $\mathbf{B}$ has linearly independent columns, $\mathbf{B}^T\mathbf{B}$ is invertible (see [appendix](#appendix) for details). Therefore, $$\mathbf{\lambda} = (\mathbf{B}^T\mathbf{B})^{-1}\mathbf{B}^T\mathbf{x}$$ Since $\pi_U(\textbf{x}) = \mathbf{B\lambda}$, we finally get $$\pi_U(\textbf{x}) = \mathbf{B}(\mathbf{B}^T\mathbf{B})^{-1}\mathbf{B}^T\mathbf{x}$$ The projection matrix $\mathbf{P_{\pi}}$ follows as $$\mathbf{P_{\pi}} = \mathbf{B}(\mathbf{B}^T\mathbf{B})^{-1}\mathbf{B}^T$$ This matrix will allow us to project any vector $\mathbf{x}$ in $\mathbf{R^n}$ and project it to any subspace $U$, using its basis vectors $\mathbf{b_1}, \dots, \mathbf{b_m}$. The intuition behind its property to stay invariant under multiple transformations is that once a vector $\mathbf{x}$ is projected onto $U$, $\mathbf{P_{{\pi}}x} \in U$. When we apply the projection again, the closest vector in $U$ to $\mathbf{P_{{\pi}}x}$ is itself so we simply project to $\mathbf{P_{{\pi}}x}$ again.

One thing to note is that when $\bf{B}$ are an orthonormal basis, $\bf{B}$ will be an orthogonal matrix, meaning that $\bf{B}^T\bf{B} = I$. This simplifies our projection matrix to be $$\mathbf{P_{\pi}} = \mathbf{B}\mathbf{B}^T$$
<br><br><br>

### Gram-Schmidt Orthogonalization

*The Gram-Schmidt Orthogonalization is a method to turn any basis that spans* $R^n$ *into a set of orthogonal basis that spans* $R^n$.

Given any set of basis vectors $(\mathbf{b_1}, \dots, \mathbf{b_m})$, we can iteratively construct a set of orthogonal/orthonormal basis vectors $(\mathbf{u_1}, \dots, \mathbf{u_m})$ for that vector space. We start by defining $$\mathbf{u_1} \coloneqq \mathbf{b_1}$$ We will now build the rest of our orthogonal vectors around this arbitrary vector. Conceptually, what we will do is project our next vector $\bf{b_2}$ onto $\mathbf{u_1}$, which will give us the ***parallel component*** of $\bf{b_2}$ with respect to $\mathbf{u_1}$. Since orthogonal vectors have no parallel component to each other, we will subtract the parallel component of $\bf{b_2}$ from $\bf{b_2}$ to get a new orthogonal vector $\bf{u_2}$. $$\bf{u_2} = \bf{b_2} - \bf{b_2}^\parallel = \bf{b_2} - \bf{\pi_{u_1}}(\bf{b_2})$$ For $\bf{b_3}$ we will subtract the parallel component of $\bf{b_3}$ with respect to both $\bf{u_1}$ and $\bf{u_2}$ from $\bf{b_3}$ to get $$\bf{u_3} = \bf{b_3} - \bf{\pi_{span[u_1, u_2]}(b_3)}$$ We can generate a rule for any $\bf{u_k}$ as $$\bf{u_k} = \bf{b_k} - \bf{\pi_{span[u_1, \dots, u_{k-1}]}(b_k)}$$ Once we repeat until $\bf{u_m}$, we have successfully built a set of orthogonal vectors $(\mathbf{u_1}, \dots, \mathbf{u_m})$. <br>
If we want to turn $(\mathbf{u_1}, \dots, \mathbf{u_m})$ into an orthonormal basis ($\lvert\mathbf{u_k}\rvert=1$), we can simply normalize each vector $\mathbf{u_k}$ as follows: $$\hat{\mathbf{u_k}} = \frac{\mathbf{u_k}}{\lvert\mathbf{u_k}\rvert}$$


<br><br><br>

### Cholesky Decomposition

*The Cholesky Decomposition decomposes a symmetric positive definite matrix into a lower triangular matrix and its transpose.*

Let $\mathbf{S} = \mathbf{A}^T\mathbf{A}$ where $\bf{A}$ has linearly independent columns, be a symmetric positive definite matrix. We need this to be the case because if we multiply out the Cholesky Decomposition, we find that we must divide certain terms by the diagonal elements. Positive semidefinite means that for any nonzero vector $\mathbf{x}$, $$\mathbf{x}^T\mathbf{A}^T\mathbf{Ax} \geq 0$$ which simplifies to $$(\mathbf{Ax})^T(\mathbf{Ax}) \geq 0$$ This is also the dot product of vector $\mathbf{Ax}$ with itself. We know that this dot product can equal $0$ whenever $\mathbf{x}$ is in the nullspace of $\mathbf{A}$, which will give us $(\mathbf{0})^T(\mathbf{0}) = 0$. Choosing one of the standard basis vectors as $\mathbf{x}$, we can get a diagonal term of $\mathbf{S}$. If one of the standard basis vectors happen to be in the nullspace of $\mathbf{A}$, this results in the diagonal term of $\mathbf{S}$ becoming $\mathbf{0}$, which will cause an error due to division by $0$.

A reason why the Cholesky Decomposition is useful is it speeds up the calculation of determinants. Since $\mathbf{S} = \mathbf{LL}^T$, $$det(\mathbf{S}) = det(\mathbf{L}) * det(\mathbf{L}^T)$$ We know that for a triangular matrix, the determinant can be calculated by multiplying the diagonal terms of the matrix. Since $det(\mathbf{L}) = det(\mathbf{L}^T)$, we can say $$det(\mathbf{S}) = det(\mathbf{L})^2$$ Therefore the determinant of $\mathbf{S}$ can be calculated by multiplying the square of the diagonal terms on its triangular matrix.

<br><br><br>

### Diagonalization

*Diagonal matrices are matrices will nonzero entries on its diagonal and zeros everywhere else. They are very useful for certain operations such as squaring the matrix, finding its inverse, or calculating its determinant.*

Obviously, most matrices you see will not be in a diagonal form. However, there are special matrices that can be turned into a diagonal matrix. More specifically these matrices are ***similar*** to diagonal matrix. A matrix $\bf{A}$ is similar to diagonal matrix $\bf{D}$ if there exists matrix $\bf{P} \in \bf{R^{nxn}}$ where $$\bf{D} = \bf{P}^{-1}\bf{AP}$$ In other words, $\bf{A}$ must be written as a change of basis to $\bf{D}$. Omitting the proof, In order for a matrix $\bf{A}$ to be diagonalizable, the columns of $\bf{P}$, $[\mathbf{p_1}, \dots, \mathbf{p_m}]$ must be the *linearly independent* eigenvectors of $\bf{A}$, and the entries of $\bf{D}$ are its corresponding eigenvalues $\lambda_i$. This tells us that matrix $\bf{A}$ must be ***nondefective*** meaning that it has $n$ linearly independent eigenvectors. Otherwise, $\bf{P}$ will not be full rank, meaning that it would not be invertible.

<br><br><br>

### Eigendecomposition

*Eigendecomposition decomposes a nondefective matrix into a matrix with eigenvectors as its columns, the inverse, and a diagonal matrix with its eigenvalues as entries.*

A direct result of diagonalization is that we can write a nondefective matrix $\bf{A}$ in terms of three simpler matrices. $$\bf{D} = \bf{P}^{-1}\bf{AP} \Longleftrightarrow \bf{PD} = \bf{AP} \Longleftrightarrow \bf{A} = \bf{PD}\bf{P}^{-1}$$ Recall that $[\mathbf{p_1}, \dots, \mathbf{p_m}]$ are the eigenvectors of $\bf{A}$, and $\bf{D}$ is a diagonal matrix with corresponding eigenvalues ${\lambda_i}$. <br>
From the spectral theorem, we know that for a symmetrical matrix, its ***eigenspaces corresponding to distinct eigenvalues are orthogonal***. Using Gram-Schmidt Orthogonalization, we can always turn those eigenvectors into an orthonormal basis. Once $\bf{P}$ is an orthogonal matrix we can simplify to $$\bf{A} = \bf{PD}\bf{P}^{T}$$ Therefore, ***any symmetric matrix has an eigendecomposition of an orthogonal eigenbasis***. 

Now I will explain an intuitive approach to eigendecomposition. For easier understanding, I will let $\bf{P}$ be an orthogonal matrix. Whenever we see a series of matrices being multiplied to each other, we can think of it as a series of linear transformations being applied from the standard basis one after another from right to left. Starting from the rightmost transformation, $\bf{P}^T = \bf{P}^{-1}$, this matrix takes the eigenvectors of $\bf{A}$, and rotates them to align with the standard basis. Once our eigenvectors are aligned with the standard basis, we will multiply by the diagonal matrix of eigenvalues. Due to the properties of diagonal matrices, the transformation will only affect each basis vector, and scale them by a factor of its corresponding eigenvalues. Now that our eigenvectors are properly scaled, we will rotate them back to their respective positions by multiplying by $\bf{P}$.

<br><br><br>

### Singular Value Decomposition

*Singular Value Decomposition (SVD) decomposes any mxn matrix into three matrices: the left singular values, singular value, and right singular values.*

Unlike the Cholensky Decomposition or Eigendecomposition, this decomposition is universal in the sense that we can apply it to any $m$x$n$ matrix. SVD is decomposed into $\bf{A} = \bf{U \Sigma V^T}$, where $\bf{U}$ is an $m$x$m$ orthogonal matrix, $\bf{V}$ is an $n$x$n$ orthogonal matrix, and $\bf{\Sigma}$ is an $m$x$n$ matrix with its diagonal values being the *singular values* of $\bf{A}$, and the remaining entries being $0$.

For an intuitive idea of the SVD, we can imagine $\bf{V^T}$ as a rotation, turning the orthogonal eigenvectors of matrix $\bf{A}$ to align with the standard basis, much like Eigendecomposition. Now, instead of simply stretching or shrinking A by its corresponding eigenvalues, it also adds new dimensions/removes dimensions from A to map it onto $R^n$. Once this is done, we once again rotate the eigenvectors back to their respective directions, much like Eigendecomposition, completing the transformation.





<br><br><br>
## Appendix

If $\mathbf{B}$ has linearly independent columns, $\mathbf{B}^T\mathbf{B}$ is invertible. <br>
We will prove $Nul\space\mathbf{B}^T\mathbf{B} = Nul\space\mathbf{B} = {\mathbf0}$. $$\mathbf{B}^T\mathbf{Bx} = {\mathbf0} \Longleftrightarrow \mathbf{x}^T\mathbf{B}^T\mathbf{Bx} = {\mathbf0} \Longleftrightarrow (\mathbf{Bx})^T\mathbf{Bx} = 0 \Longleftrightarrow \langle\mathbf{Bx}, \mathbf{Bx}\rangle = \mathbf{0} \Longleftrightarrow \mathbf{Bx} = {\mathbf{0}}$$ Here, we utilize the fact that if the dot product of a vector by itself equals $0$, by the positive definite property of the dot product the vector must be $\mathbf{0}$.
