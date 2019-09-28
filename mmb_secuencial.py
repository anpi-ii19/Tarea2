from sympy import MatrixSymbol, Symbol, lambdify, diff
from numpy import matrix, ones, zeros, linalg
from scipy import optimize
from graficar_error import graficar_error


def mmb_secuencial(n, tol):
    """
    Metodo de Mejora Maxima por Bloque
    :param n: Cantidad de filas y columnas de la matriz tridiagonal
    :param tol: Tolerancia al fallo que debe tener el resultado
    :return: Matriz x calculada con el metodo
    """
    if n <= 1:
        return "El parametro n debe ser mayor a 1"

    # Se crea la matriz tridiagonal y la matriz de resultados
    matriz_a = crear_matriz_a(n)
    matriz_b = crear_matriz_b(n)

    # Se crea el vector inicial
    vec_x = matrix(ones((n, 1), dtype='object'))

    # Se crean las variables simbolicas
    simbolos = [MatrixSymbol('A', n, n),
                MatrixSymbol('b', n, 1),
                MatrixSymbol('x', n, 1)]

    simb_a = simbolos[0]  # Variable simbolica de la matriz A
    simb_b = simbolos[1]  # Variable simbolica de la matriz b
    simb_x = simbolos[2]  # Variable simbolica de la matriz x

    # Se crea la funcion que debe ser evaluada
    f = 1/2 * simb_x.T * simb_a * simb_x - simb_b.T * simb_x
    funcion = lambdify(simbolos, f)  # f(A, b, x)

    z = Symbol('z')

    vec_x_simbolico = []
    # Se crean todas las variables simbolicas del vector
    for i in range(0, n):
        vec_x_simbolico.append([Symbol('x' + str(i))])

    # Se construye una np.matrix a partir del vector obtenido
    vec_x_simbolico = matrix(vec_x_simbolico)

    gradiente = calcular_gradiente(funcion, matriz_a, matriz_b, vec_x_simbolico)

    # Se crean los vectores para las iteraciones
    vec_x1_aux = matrix(zeros((n, 1), dtype='object'))
    vec_error = matrix(zeros((n, 1), dtype='object'))

    # Se crean las listas para graficar el error
    lista_iter = []
    lista_error = []

    itr = 1  # Contador de iteraciones

    while 1:
        # Se calcula el vector k+1 utilizando la regla de Jacobi
        for j in range(0, n):
            copia_vec_x = vec_x.copy()
            copia_vec_x[j] = z

            # Se evalua la funcion con el vec_x con la variable simbolica
            ecuacion = funcion(matriz_a, matriz_b, copia_vec_x)

            # Se crea la funcion a minimizar
            fun_minimizar = lambdify(z, ecuacion.item(0))

            # Se minimiza la funcion
            resultado = optimize.minimize_scalar(fun_minimizar).x

            # Se actualiza el vector de la iteracion actual
            vec_x1_aux[j] = resultado

            # Se calcula el valor para el vector de error
            copia_vec_x[j] = resultado
            vec_error[j] = funcion(matriz_a, matriz_b, copia_vec_x)

        # Se obtiene la posicion del menor elemento en el vector de error
        i = vec_error.argmin()

        # Se actualiza el valor en la posicion minima
        vec_x[i] = vec_x1_aux[i]

        # Se calcula el vector para verificar la condicion de parada
        vec_parada = evaluar_gradiente(gradiente, vec_x_simbolico, vec_x)

        norma_2 = linalg.norm(vec_parada, 2)

        lista_iter.append(itr)
        lista_error.append(norma_2)

        # Se verifica la condicion de parada
        if norma_2 < tol:
            break

        itr += 1

    graficar_error(lista_iter, lista_error)

    return vec_x


def calcular_gradiente(f, matriz_a, matriz_b, vec_x_simbolico):
    """
    Calculo del gradiente de una funcion matematica
    :param f: Funcion lambdify a la que calcularle el gradiente
    :param matriz_a: Matriz A
    :param matriz_b: Matriz b
    :param vec_x_simbolico: Vector x con todas las variables simbolicas
    :return: Gradiente calculado
    """
    gradiente = []
    m = matriz_b.shape[0]

    # Se evalua la funcion con el vector simbolico
    funcion = f(matriz_a, matriz_b, vec_x_simbolico).item(0)

    # Se calcula el gradiente de la funcion obtenida
    for j in range(0, m):
        df = diff(funcion, vec_x_simbolico[j]).tolist()[0][0]

        gradiente.append(df)

    return gradiente


def evaluar_gradiente(gradiente, variables, vector):
    """
    Funcion para evaluar el gradradiente con un vector ingresado
    :param gradiente: gradiente que se debe evaluar
    :param variables: lista con las variables simbolicas de la ecuacion
    :param vector: vector que se debe evaluar en el gradiente
    :return: resultado de evaluar el vector en el gradiente
    """
    n = len(variables)
    resultado = []

    # Se recorre cada una de las derivadas parciales en el gradiente
    for i in range(0, n):
        # Se obtiene la derivada parcial
        funcion = gradiente[i]

        # Se sustituyen cada una de las variables por el valor en el vector
        for x in range(0, n):
            funcion = funcion.subs(variables.item(x), vector.item(x))

        resultado += [[funcion.doit()]]

    return matrix(resultado, dtype='float')


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


# n = 10
# r = mmb_secuencial(n, 10**-5)
# a = crear_matriz_a(n)
# v = a * r
