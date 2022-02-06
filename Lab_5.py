import numpy as np


def power_method(A):
    eigvals = []
    eigvects = []
    B = A.copy()

    for i in range(A.shape[0]):
        eigenvalue, eigenvector = power_iteration(B)

        eigvals.append(eigenvalue)
        eigvects.append(eigenvector)

        if i < A.shape[0]:
            B -= eigenvalue * np.outer(eigenvector, eigenvector)

    return eigvals, eigvects


def power_iteration(A, epsilon=1e-10, max_iter=100):
    z = np.random.rand(A.shape[0])

    for i in range(max_iter):
        w = A @ z
        l = max(np.abs(w))
        if np.linalg.norm(w - l * z) < epsilon:
            break
        z = w / l

    eigval = l
    eigvect = z / np.linalg.norm(z)

    return eigval, eigvect


def svd(A):
    eigvals, eigvectsAAt = power_method(A @ A.T)
    eigvectsAtA = power_method(A.T @ A)[1]
    singular_values = np.sqrt(eigvals)

    U = np.column_stack(eigvectsAAt)

    Sigma = np.diag(singular_values)

    V = np.column_stack(eigvectsAtA)

    return U, Sigma, V


if __name__ == '__main__':
    from_lecture = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]]).astype(float)

    U, Sigma, V = svd(from_lecture)
    print("Matrix:\n", from_lecture)
    print("========================")
    print("U:\n", U)
    print("SIGMA:\n", Sigma)
    print("V:\n", V)

    # A = np.array([[3, 1], [6, 2]], dtype=np.float32)
    #
    # U, Sigma, V = svd(A)
    # print("Matrix:\n", A)
    # print("========================")
    # print("U:\n", U)
    # print("SIGMA:\n", Sigma)
    # print("V:\n", V)





    # # result = U @ Sigma @ V.T
    # # print(result)

