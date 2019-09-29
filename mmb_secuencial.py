from sympy import MatrixSymbol, Symbol, lambdify
from numpy import matrix, ones, zeros, linalg
from scipy import optimize
from graficar_error import graficar_error
from crear_matriz import crear_matriz_a
from crear_matriz import crear_matriz_b
from gradiente import calcular_gradiente
from gradiente import evaluar_gradiente


def mmb_secuencial(n, tol):
    """
    Metodo de Mejora Maxima por Bloque utilizando una ejecucion secuencial
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
        vec_parada = matrix(vec_parada, dtype='float')

        norma_2 = linalg.norm(vec_parada, 2)

        lista_iter.append(itr)
        lista_error.append(norma_2)

        # Se verifica la condicion de parada
        if norma_2 < tol:
            break

        itr += 1

    graficar_error(lista_iter, lista_error)

    return vec_x


# n = 10
# r = mmb_secuencial(n, 10**-5)
# a = crear_matriz_a(n)
# v = a * r
