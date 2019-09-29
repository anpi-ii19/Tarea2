import time
import numpy as np
from scipy.optimize import fmin

def create_A(n):
    up =np.diag(np.ones(n-1)*2,1)
    down =np.diag(np.ones(n-1)*2,-1)
    center=np.diag(np.ones(n)*6,0)
    return np.matrix(up+down+center, dtype='float')

def create_b(n):
    b = np.ones(n)*15
    b[-1]=12
    b[0]=12
    return np.matrix(b, dtype='float').T

def create_x0(n):
    return np.matrix(np.ones(n), dtype='float').T

def fun(A,b,x):
    return ((1/2)*x.T*A*x-b.T*x).item()

def grad_fun(X,A,b):
    x_size=len(X)
    grad = np.zeros(x_size)
    for i in range(0,x_size):
        grad[i]= (1/2) * ( A.T[i]*X+A[i]*X)-(b[i])
    return grad


def jacobi(f,X_0):
    x_size=len(X_0)
    results=np.zeros(x_size)
    for i in range(0,x_size):
        f_i = lambda x :f_n(f,X_0,i,x)
        results[i]=fmin(f_i,X_0[i],disp=False)
    return np.matrix(results).T

def f_n(f,X,n,x):
    X_local= np.copy(X)
    X_local[n]=x
    return f(X_local)
    

def mmb(f, grad_f,x_0,tol=10**-5, disp=False):
    x_k=np.copy(x_0)
    x_size= len(x_k)
    cont=0
    norm_grad_prev=np.linalg.norm(grad_f(x_k))
    while(  norm_grad_prev >tol ):
        e=np.zeros(x_size)
        x_tem=np.zeros(x_size)
        for i in range(x_size):
            f_i = lambda x :f_n(f,x_k,i,x)
            x_min = fmin(f_i,x_k[i],disp=False,ftol=10**-15)
            x_tem[i]=x_min
            x_k_tem = np.copy(x_k)
            x_k_tem[i]=x_min
            e[i]=f(x_k_tem)
        i = e.argmin()
        x_k[i]=x_tem[i]
        norm_grad_current=np.linalg.norm(grad_f(x_k))
        if(abs(norm_grad_current-norm_grad_prev)<10**-30):
            print("BREAK")
            break
        if(disp):
#             print(cont,"error:",np.linalg.norm(grad_f(x_k)))
            print(cont,norm_grad_current,abs(norm_grad_current- norm_grad_prev))
        norm_grad_prev=norm_grad_current
        
        cont+=1
    return x_k

def run_mmb(n, tol=10**-5,disp=False):
    A=create_A(n)
    b=create_b(n)
    x_0=create_x0(n)
    f = lambda x: fun(A,b,x)
    grad_f = lambda x:  grad_fun(x,A,b)
    start = time.time()
    result = mmb(f,grad_f,x_0,tol,disp=disp)
    end = time.time()
    delta=end-start
    print("-"*100)
    print("time:",delta)
    print("-"*100)
    return result

# run_mmb(50,disp=True)
