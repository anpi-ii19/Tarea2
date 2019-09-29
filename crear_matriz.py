from numpy import matrix, zeros


def crear_matriz_a(n):
    """
    Funcion encargada de crear la matriz tridiagonal A
    :param n: Numero de filas y columnas de la matriz
    :return:
    """
    # Se crea una matriz de n x n
    matriz_a = matrix(zeros((n, n), dtype='int'))

    # Caso especial primera fila
    matriz_a[0, 0] = 6
    matriz_a[0, 1] = 2

    # Se itera sobre las filas de la matriz para establecer los valores
    for i in range(1, n - 1):
        matriz_a[i, i] = 6
        matriz_a[i, i - 1] = 2
        matriz_a[i, i + 1] = 2

    # Caso especial ultima fila
    matriz_a[n - 1, n - 1] = 6
    matriz_a[n - 1, n - 2] = 2

    return matriz_a


def crear_matriz_b(n):
    """
    Funcion encargada de crear la matriz b de una columna
    :param n: Numero de filas
    :return: Matriz creada
    """
    # Se crea un vector columna de n x 1
    matriz_b = matrix(zeros((n, 1), dtype='int'))

    # Caso especial primera fila
    matriz_b[0, 0] = 12

    # Se itera sobre las filas de la matriz para establecer los valores
    for i in range(1, n - 1):
        matriz_b[i, 0] = 15

    # Caso especial ultima fila
    matriz_b[n - 1, 0] = 12

    return matriz_b
