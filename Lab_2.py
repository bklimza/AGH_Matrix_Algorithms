import numpy as np


def Gauss_without_pivot_with_ones(A):
    A = np.array(A, dtype=float)
    m = A.shape[0]
    n = m + 1  # index from zero

    for i in range(0, m):  # row
        div = A[i, i]
        for s in range(n):
            A[i, s] = A[i, s] / div

        for j in range(i + 1, m):
            factor = A[j, i] / A[i, i]
            for k in range(i + 1, n):  # column
                A[j, k] -= factor * A[i, k]
                A[j, k] = A[j, k] / A[j, i]
            A[j, i] = 0

    return generate_x(A)


def Gauss_without_pivot_without_ones(A):
    A = np.array(A, dtype=float)
    m = A.shape[0]
    n = m + 1  # index from zero

    for i in range(0, m):  # row
        for j in range(i + 1, m):
            factor = A[j, i] / A[i, i]
            for k in range(i, n):  # column
                A[j, k] -= factor * A[i, k]
            A[j, i] = 0

    return generate_x(A)


def Gauss_with_pivot(A):
    A = np.array(A, dtype=float)
    m = A.shape[0]
    n = m + 1  # index from zero

    for i in range(0, m):  # row
        for s in range(i, m):
            pivot = [abs(A[s, i])]
            max_i = pivot.index(max(pivot)) + i
            A[i], A[max_i] = A[max_i], A[i]

            for j in range(i + 1, m):
                factor = A[j, i] / A[i, i]
                for k in range(i + 1, n):  # column
                    A[j, k] -= factor * A[i, k]
                A[j, i] = 0

    return generate_x(A)


def generate_x(A):
    A = np.array(A, dtype=float)
    m = A.shape[0]
    x = []
    for j in range(m - 1, -1, -1):
        x.insert(0, round(A[j, m] / A[j, j], 3))
        for i in range(j - 1, -1, -1):
            A[i, m] -= A[i, j] * x[0]
    return x


def LU_factorization(A):
    m = len(A)
    L = np.array(A, dtype='f')
    U = np.array(A, dtype='f')
    # Upper
    for i in range(0, m):  # row
        for j in range(i + 1, m):
            factor = int(A[j][i] / A[i][i])
            for k in range(i, m):  # column
                U[j][k] = U[j][k] - factor * A[i][k]
            U[j][i] = 0
    # Lower
    for i in range(0, m):
        for k in range(i, m):
            if i == k:
                L[i][i] = 1
            elif U[i][i] != 0:
                sum_ = 0
                for j in range(i):
                    sum_ += (L[k][j] * U[j][i])

                L[k][i] = int((A[k][i] - sum_) / U[i][i])
                L[i][k] = 0
            else:
                pass

    # print L, matrix_u
    print("Lower=  \t\tUpper=")

    # Displaying the result :
    for i in range(m):
        # Lower
        for j in range(m):
            print(int(L[i][j]), end="\t")
        print("", end="\t\t")
        # Upper
        for j in range(m):
            print(int(U[i][j]), end="\t")
        print("")


A = np.array([[999, 998, 1997], [1000, 999, 1999]])
A1 = np.array([[2, -1, 1, -4], [8, 2, 5, -10], [4, 1, 1, 2]])

a = Gauss_without_pivot_with_ones(A)
a1 = Gauss_without_pivot_with_ones(A1)
b = Gauss_without_pivot_without_ones(A)
b1 = Gauss_without_pivot_without_ones(A1)
c = Gauss_with_pivot(A)
c1 = Gauss_with_pivot(A1)

print(a)
print(a1)

A_LU = [[999, 998],
        [1000, 999]]
A_LU1 = [[2, -1, 1], [8, 2, 5], [4, 1, 1]]
LU_factorization(A_LU1)
