import numpy as np
from numpy.linalg import norm, det
import random


def power_method(A, iteration, epsilon=1e-1):
    n = A.shape[0]
    z = np.array([random.randint(-1, 1) for _ in range(n)])
    while True:
        if A.shape[0] == 2 and iteration == 2 and det(A) < epsilon:
            return [A[0][1] / A[0][0], -A[0][0] / A[0][0]], 0

        w = np.dot(A, z)
        l = max(abs(w))
        if norm(w - l * z) < epsilon:
            break
        z = w / l
    return z, l


def iterate_power_method(A):
    n = A.shape[0]
    lambda_list = []
    z_list = []
    for i in range(n):
        z, l = power_method(A, i + 1)
        if norm(z) == 0:
            continue
        lambda_list.append(l)
        z_list.append(z / norm(z))
        w = np.dot(A, z)

        product = np.outer(w, w.T)
        A = A - product
    return lambda_list, z_list


def svd(A):
    n = A.shape[0]
    B = np.dot(A, A.T)
    B_z_list, B_lambda_list = iterate_power_method(B)
    singular_values = np.sqrt(B_z_list)
    sigma = np.empty((n, n))
    for i in range(n):
        sigma[i][i] = singular_values[i]

    C = np.dot(A.T, A)
    C_z_list, C_lambda_list = iterate_power_method(C)

    U = np.array(B_lambda_list)
    sigma = np.array(sigma)
    V = np.array(C_lambda_list)
    return U, sigma, V.T


if __name__ == "__main__":
    A = np.array([[3, 1], [6, 2]])

    U, sigma, V_T = svd(A)
    # result = U @ sigma @ V_T
    print(A)
    print("U:")
    print(U)
    print("sigma:")
    print(sigma)
    print("V.T:")
    print(V_T)
    # print(result)
