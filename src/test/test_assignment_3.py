import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'main')))

import numpy as np
from assignment_3 import (
    gaussian_elimination_with_back_substitution,
    lu_factorization,
    is_diagonally_dominant,
    is_positive_definite
)

# Q1: Gaussian Elimination Test
matrix_q1 = np.array([
    [1, 1, 0, 3],
    [2, 1, -1, 1],
    [3, -1, -1, 2]
], dtype=float)
result_q1 = gaussian_elimination_with_back_substitution(matrix_q1.copy())
print("Q1 - Gaussian Elimination Result:", result_q1)

# Q2: LU Factorization Test
matrix_q2 = np.array([
    [1, 1, 0, 3],
    [2, 1, -1, 1],
    [3, -1, -1, 2],
    [-1, 2, 3, -1]
], dtype=float)
L, U = lu_factorization(matrix_q2.copy())
determinant_q2 = np.prod(np.diag(U))
print("Q2 - Determinant:", determinant_q2)
print("Q2 - L Matrix:\n", L)
print("Q2 - U Matrix:\n", U)

# Q3: Diagonal Dominance Test
matrix_q3 = np.array([
    [9, 0, 5, 2, 1],
    [3, 9, 1, 2, 1],
    [0, 1, 7, 2, 3],
    [4, 2, 3, 12, 2],
    [3, 2, 4, 0, 8]
], dtype=float)
result_q3 = is_diagonally_dominant(matrix_q3)
print("Q3 - Diagonally Dominant:", result_q3)

# Q4: Positive Definiteness Test
matrix_q4 = np.array([
    [2, 2, 1],
    [2, 3, 0],
    [1, 0, 2]
], dtype=float)
result_q4 = is_positive_definite(matrix_q4)
print("Q4 - Positive Definite:", result_q4)
