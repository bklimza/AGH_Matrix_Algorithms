import math
import numpy as np


def check_norm(A, p):
    result = get_matrix_norm(A, p)
    correct_norm_A = np.linalg.norm(A, p)
    print(f'Norm of A: {result}')
    print(f'Correct norm: {correct_norm_A}')
    return math.isclose(correct_norm_A, result, rel_tol=1e-1)


def get_matrix_norm(A, p):
    print(A, "\n")
    print("x = ", x, "\n")
    norm_x = np.sum(np.abs(x) ** p, axis=0) ** (1 / p)
    Ax = A.dot(x)
    norm_Ax = np.sum(np.abs(Ax) ** p, axis=0) ** (1 / p)
    norm_A = np.max(norm_Ax / norm_x)
    return round(norm_A, 2)


def check_inverse_matrix(A):
    print("A = ", A)
    B = np.linalg.inv(A)
    return B


def get_matrix_minor(M, i, j):
    A = np.ndarray.tolist(M)
    return [row[:j] + row[j+1:] for row in (A[:i]+A[i+1:])]

def get_inverse_matrix(A):
    determinant = np.linalg.det(A)
    if determinant != 0:
        #for 2x2 matrix:
        if len(A) == 2:
            return np.array([[A[1][1] / determinant, -1 * A[0][1] / determinant],
                    [-1 * A[1][0] / determinant, A[0][0] / determinant]])

        cofactors = []
        for r in range(len(A)):
            cofactorRow = []
            for c in range(len(A)):
                minor = get_matrix_minor(A, r, c)
                cofactorRow.append(((-1)**(r+c)) * np.linalg.det(minor))
            cofactors.append(cofactorRow)
        cofactors_matrix = np.array(cofactors)
        cofactors_matrix_t = cofactors_matrix.T
        for r in range(len(cofactors_matrix_t)):
            for c in range(len(cofactors_matrix_t)):
                cofactors_matrix_t[r][c] = cofactors_matrix_t[r][c]/determinant
        return np.array(cofactors_matrix_t)
    else:
        print("Determinant equals zero!")


if __name__ == "__main__":
    A_from_lecture = np.array([[4, 9, 2], [3, 5, 7], [8, 1, 6]])
    # x = np.array([1, 1, 1])
    print(A_from_lecture)
    print(get_inverse_matrix(A_from_lecture))

    sizes = [2, 3]
    for n in sizes:
        A = np.random.randint(low=1, high=10, size=(n, n)).astype('float32')
        x = np.random.randint(low=1, high=10, size=(A.shape[0], 1000)).astype('float32')

        print(check_norm(A, 1))
        print("Norm-1 of matrix A: ", get_matrix_norm(A, 1))
        print("Norm-2 of matrix A: ", get_matrix_norm(A, 2))
        print("Norm-3 of matrix A: ", get_matrix_norm(A, 3))
        C = get_inverse_matrix(A)
        print("Norm-1 of matrix C: ", get_matrix_norm(C, 1))
        print("Norm-2 of matrix C: ", get_matrix_norm(C, 2))
        print("Norm-3 of matrix C: ", get_matrix_norm(C, 3))
