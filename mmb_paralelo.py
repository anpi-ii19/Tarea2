from sympy import MatrixSymbol, Symbol, lambdify
from numpy import matrix, ones, linalg
from scipy import optimize
from graficar_error import graficar_error
from crear_matriz import crear_matriz_a
from crear_matriz import crear_matriz_b
from gradiente import calcular_gradiente
from gradiente import evaluar_gradiente
from time import time
from multiprocessing import Process, Queue


def mmb_paralelo(n, tol):
    """
    Metodo de Mejora Maxima por Bloque utilizando paralelismo
    :param n: Cantidad de filas y columnas de la matriz tridiagonal
    :param tol: Tolerancia al fallo que debe tener el resultado
    :return: Matriz x calculada con el metodo
    """
    start_time = time()

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

    vec_x_simbolico = []
    # Se crean todas las variables simbolicas del vector
    for i in range(0, n):
        vec_x_simbolico.append([Symbol('x' + str(i))])

    # Se construye una np.matrix a partir del vector obtenido
    vec_x_simbolico = matrix(vec_x_simbolico)

    gradiente = calcular_gradiente(funcion, matriz_a, matriz_b, vec_x_simbolico)

    # Se crean las listas para graficar el error
    lista_iter = []
    lista_error = []

    itr = 1  # Contador de iteraciones

    # Se crea una cola para almacenar el resultado de los procesos
    cola = Queue()

    while 1:
        # start_time = time()
        # Se crea una lista para almacenar los procesos
        procesos = []

        # Se calcula el vector k+1 utilizando la regla de Jacobi
        for j in range(0, n):
            # Se crea un proceso para aplicar la regla de jacobi en la posicion j
            proceso = Process(target=jacobi, args=(funcion, matriz_a, matriz_b, vec_x.copy(), j, cola))
            procesos.append(proceso)
            proceso.start()

        # Se espera a que todos los procesos finalicen
        for proceso in procesos:
            proceso.join()

        # Se obtiene el valor que posea el menor error, se saca un elemento de la cola, y como es el
        # primero se toma como el minimo, y se va recorriendo la cola buscando elementos menores a este
        resultado = cola.get()          # Lista: [resultado, error, posicion]
        min_error_valor = resultado[0]  # Resultado que tiene el menor error
        min_error = resultado[1]        # Error minimo
        min_error_pos = resultado[2]    # Posicion del resultado

        while not cola.empty():
            resultado = cola.get()
            error = resultado[1]
            if error < min_error:
                # Se actualizan los valores minimos
                min_error = error
                min_error_valor = resultado[0]
                min_error_pos = resultado[2]

        # elapsed_time = time() - start_time
        # print(elapsed_time)

        # Se actualiza el valor en la posicion minima
        vec_x[min_error_pos] = min_error_valor

        # Se calcula el vector para verificar la condicion de parada
        vec_parada = evaluar_gradiente(gradiente, vec_x_simbolico, vec_x)
        vec_parada = matrix(vec_parada, dtype='float')

        norma_2 = linalg.norm(vec_parada, 2)

        lista_iter.append(itr)
        lista_error.append(norma_2)

        # Se verifica la condicion de parada
        if norma_2 < tol:
            break

        itr += 1

    elapsed_time = time() - start_time
    print(elapsed_time)

    graficar_error(lista_iter, lista_error)

    return vec_x


def jacobi(funcion, matriz_a, matriz_b, vec_x, j, cola):
    """
    Metodo que aplica la regla de jacobi para una posicion j
    :param funcion: Funcion que debe ser evaluada
    :param matriz_a: Matriz A
    :param matriz_b: Matriz B
    :param vec_x: vector x
    :param j: Posicion j en la que aplicar la regla de jacobi
    """
    z = Symbol('z')
    vec_x[j] = z

    # Se evalua la funcion con el vec_x con la variable simbolica
    ecuacion = funcion(matriz_a, matriz_b, vec_x)

    # Se crea la funcion a minimizar
    fun_minimizar = lambdify(z, ecuacion.item(0))

    # Se minimiza la funcion
    resultado = optimize.minimize_scalar(fun_minimizar).x

    # Se calcula el error
    vec_x[j] = resultado
    error = funcion(matriz_a, matriz_b, vec_x).item(0)

    # Se almacena en la cola una lista, donde la primera posicion es el resultado
    # obtenido, la segunda posicion es el error del resultado obtenido, y la tercera
    # posicion es el numero j
    cola.put([resultado, error, j])


# n = 11
# r = mmb_paralelo(n, 10**-5)
# a = crear_matriz_a(n)
# v = a * r
