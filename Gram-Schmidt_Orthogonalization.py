import numpy as np
from Projection import projection

def orthogonalize(basis, normalize = True):
    orth_basis = np.zeros(basis.shape)

    print(basis)

    # set the first arbitrary basis vector u_1 as b_1
    orth_basis[:, 0] = basis[:, 0]
    
    # create orthogonal vectors u_k
    for i in range(2, basis.shape[0]+1):
        orth_basis[:, i-1] = basis[:, i-1] - projection(basis[:, i-1], orth_basis[:, :i-1])

    if(normalize):
        for j in range(1, orth_basis.shape[0]+1):
            orth_basis[:, j-1] = orth_basis[:, j-1]/np.linalg.norm(orth_basis[:, j-1])
    
    return(orth_basis)


