import numpy as np
from time import perf_counter


def multiply_ijk(size, matrix_a, matrix_b):
    matrix_c = np.zeros([size, size])

    for i in range(size):
        for j in range(size):
            for k in range(size):
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return matrix_c


def multiply_ikj(size, matrix_a, matrix_b):
    matrix_c = np.zeros([size, size])

    for i in range(size):
        for k in range(size):
            for j in range(size):
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return matrix_c


def multiply_jik(size, matrix_a, matrix_b):
    matrix_c = np.zeros([size, size])

    for j in range(size):
        for i in range(size):
            for k in range(size):
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return matrix_c


def multiply_jki(size, matrix_a, matrix_b):
    matrix_c = np.zeros([size, size])

    for j in range(size):
        for k in range(size):
            for i in range(size):
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return matrix_c


def multiply_kij(size, matrix_a, matrix_b):
    matrix_c = np.zeros([size, size])

    for k in range(size):
        for i in range(size):
            for j in range(size):
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return matrix_c


def multiply_kji(size, matrix_a, matrix_b):
    matrix_c = np.zeros([size, size])

    for k in range(size):
        for j in range(size):
            for i in range(size):
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return matrix_c


def check_performance_time_ms(func, size, f, s):
    t_start = perf_counter()
    func(size, f, s)
    t_stop = perf_counter()
    return 1000 * (t_stop - t_start)


for d in [10, 100, 1000]:
    first_matrix = np.random.random((d, d))
    second_matrix = np.random.random((d, d))
    methods = {multiply_ijk: "multiply_ijk", multiply_ikj: "multiply_ikj", multiply_jik: "multiply_jik",
               multiply_jki: "multiply_jki", multiply_kij: "multiply_kij", multiply_kji: "multiply_kji"}
    for method in methods.keys():
        print(methods[method], d)
        print(method(d, first_matrix, second_matrix), "\n")
        print(check_performance_time_ms(method, d, first_matrix, second_matrix), "ms", "\n")
