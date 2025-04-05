import numpy as np

def gaussian_elimination_with_back_substitution(matrix):
    n = len(matrix)
    
    for i in range(n):
        max_index = np.argmax(np.abs(matrix[i:, i])) + i
        if i != max_index:
            matrix[[i, max_index]] = matrix[[max_index, i]]
        
        for j in range(i+1, n):
            factor = matrix[j][i] / matrix[i][i]
            matrix[j, i:] = matrix[j, i:] - factor * matrix[i, i:]
    
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (matrix[i][-1] - np.dot(matrix[i, i+1:n], x[i+1:n])) / matrix[i][i]

    return x

def lu_factorization(matrix):
    n = len(matrix)
    L = np.eye(n)
    U = matrix.copy()

    for i in range(n):
        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]
            U[j, i:] -= factor * U[i, i:]
            L[j, i] = factor

    return L, U

def is_diagonally_dominant(matrix):
    for i in range(len(matrix)):
        row_sum = sum(abs(matrix[i, j]) for j in range(len(matrix)) if j != i)
        if abs(matrix[i, i]) < row_sum:
            return False
    return True

def is_positive_definite(matrix):
    if not np.allclose(matrix, matrix.T):
        return False
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False
