from sympy import diff


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
    :return: lista python con el resultado de evaluar el vector en el gradiente
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

    return resultado
