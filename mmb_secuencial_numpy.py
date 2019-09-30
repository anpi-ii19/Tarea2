import time
import numpy as np
from scipy.optimize import fmin
from graficar_error import graficar_error


def create_A(n):
    """
    crear matrix triagiagonal A de tamano nxn
    :param n: Cantidad de filas y columnas de la matriz tridiagonal
    :return: Matriz generada
    """
    up = np.diag(np.ones(n-1)*2, 1)
    down = np.diag(np.ones(n-1)*2, -1)
    center = np.diag(np.ones(n)*6, 0)
    return np.matrix(up+down+center, dtype='float')


def create_b(n):
    """
    crear vector columna b de tamano n
    :param n: Tamano de b 
    :return: Vector columna generado
    """
    b = np.ones(n)*15
    b[-1] = 12
    b[0] = 12
    return np.matrix(b, dtype='float').T


def create_x0(n):
    """
    crear vector columna x de tamano n
    :param n: Tamano de x_0 
    :return: Vector columna generado
    """
    return np.matrix(np.ones(n), dtype='float').T


def fun(A, b, x):
    """
    Funcion (1/2)*transpuesta(x)*A*x-transpuesta(b)*x
    :param A: matriz de tamano n*n
    :param b: vector columna de tamano n
    :param x: vector columna de tamano n
    :return: (1/2)*transpuesta(x)*A*x-transpuesta(b)*x
    """
    return ((1/2)*x.T*A*x-b.T*x).item()


def grad_fun(X, A, b):
    """
    Gradiente de la funcion (1/2)*transpuesta(x)*A*x-transpuesta(b)*x, evaluada para un punto X
    :param A: matriz de tamano n*n
    :param b: vector columna de tamano n
    :param x: vector columna de tamano n para el que se evalua la gradiente
    :return: gradiente((1/2)*transpuesta(x)*A*x-transpuesta(b)*x) evaluado en X
    """

    x_size = len(X)
    grad = np.zeros(x_size)
    for i in range(0, x_size):
        grad[i] = (1/2) * (A.T[i]*X+A[i]*X)-(b[i])
    return grad


def f_n(f, X, n, x):
    """
    Llamar la funcion f actualizando unicamente la posicion n del vector X
    :param X: vector columna de tamano n
    :param n: indice de la posicion a reemplazar
    :param x: valor a colocar en X[n]
    :return: resultado de f evaluada en el vector actualizado
    """
    X_local = np.copy(X)
    X_local[n] = x
    return f(X_local)


def mmb(f, grad_f, x_0, tol=10**-5,graf_error=False, disp=False):
    """
    Metodo iterativo de mejora maxima de bloque
    :param f: funcion sobre la que se evalua la optimizacion
    :param grad_f: gradiente de la funcion f
    :param x_0: valor inicia de x
    :param tol: valor de tolerancia
    :param graf_error: bandera para generar grafico de error
    :param disp: bandera para impimir datos intermedios
    :return: argmin de f(x)
    """
    # generar copia de x_0 para las siguentes iteraciones
    x_k = np.copy(x_0)
    x_size = len(x_k)
    iter_ = 0
    #evaluar la gradiente para el x_0
    norm_grad_current = np.linalg.norm(grad_f(x_k))
    error=[]
    while(norm_grad_current > tol):
        e = np.zeros(x_size)
        x_tem = np.zeros(x_size)
        for i in range(x_size):
            def f_i(x): return f_n(f, x_k, i, x)
            #encontrar el argmin para x_i
            x_min = fmin(f_i, x_k[i], disp=False, ftol=10**-15)
            x_tem[i] = x_min
            x_k_tem = np.copy(x_k)
            x_k_tem[i] = x_min
            e[i] = f(x_k_tem)
        i = e.argmin()
        x_k[i] = x_tem[i]
        norm_grad_prev = norm_grad_current
        #evaluar la gradiente en x_k
        norm_grad_current = np.linalg.norm(grad_f(x_k))
        #parada en caso de no converger
        if(abs(norm_grad_current-norm_grad_prev) < 10**-30):
            print("BREAK")
            break
        if(disp):
            print(iter_,"error:",np.linalg.norm(grad_f(x_k)))
        
        error.append(norm_grad_current)
        iter_ += 1
    if(graf_error):
        graficar_error(range(iter_),error)
    return x_k


def run_mmb(n, tol=10**-5,graf_error=True, disp=False):
    """
    Ejecutar la funcion mmb, inicianlizando los vectores con un tamano n y midiendo los tiempos de ejecucion
    :param n: tamano de los vectores y matrices
    :param tol: valor de tolerancia
    :param graf_error: bandera para generar grafico de error
    :param disp: bandera para impimir datos intermedios
    :return: argmin de f(x)
    """
    A = create_A(n)
    b = create_b(n)
    x_0 = create_x0(n)
    f= lambda x: fun(A, b, x)
    grad_f =  lambda x: grad_fun(x, A, b)
    start = time.time()
    result = mmb(f, grad_f, x_0, tol,graf_error=graf_error, disp=disp)
    end = time.time()
    delta = end-start
    print("-"*30)
    print("time:", delta)
    print("-"*30)
    return result

# x_min = run_mmb(50,disp=True)
# print(x_min)