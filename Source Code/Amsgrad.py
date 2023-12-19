# This code implements the Amsgrad optimization algorithm based on the cost function provided by the user.


import sys
import numpy as np
import scipy as sc
from sympy import Symbol, diff, lambdify, sympify


def Amsgrad(cost_function, f):
    x = Symbol('x')
    print("f(x) = ", cost_function)
    f_dash = diff(cost_function, x)
    print("df(x)/dx = ", f_dash)
    initialApproximation = float(input("\n---> Enter initial approximation: "))
    x0 = initialApproximation
    errorTolerance = float(input("---> Enter error tolerance: "))
    learningRate = float(input("---> Enter learning rate: "))
    print("\n---------------------------------------------------------------")
    print(" *** Starting Adam")
    print("        --->  x0 = ", initialApproximation)
    print("        --->  f(x0) = ", f(initialApproximation))
 
    #----------------------------------------------------------------------------------------------------------------------------------------------------
    iterationCount = 0
    xk = x0
    x_prev = 0.0
    m0 = 0.0
    mk = 0.0
    v0 = 0.0
    vk = 0.0
    vc_0 = 0.0
    vc_k = 0.0
    b1 = 0.9
    b2 = 0.999
    epsilon = 10 ** -8
    while True:
        iterationCount += 1
        x_prev = x0
        x0 = xk
        m0 = mk
        v0 = vk
        vc_0 = vc_k
        fk_dash = (lambdify(x, f_dash, "numpy"))(xk)            # Compute the derivative of f and assign it to fk_dash
        gt = fk_dash
        mk = b1 * m0 + (1 - b1) * gt                            # Update the first moment
        vk = b2 * v0 + (1 - b2) * (gt ** 2)                     # Update the second moment
        vc_k = max(vc_0, vk)
        xk = xk - (learningRate / (vc_k ** 0.5 + epsilon)) * mk
          
        if abs(N(xk - x0)) < float(errorTolerance) or abs(N(xk - x_prev)) < 0.1 * float(errorTolerance):      # Check convergence condition
            break
    #----------------------------------------------------------------------------------------------------------------------------------------------------
    
    print(" *** Number of Iterations = ", iterationCount)
    print("        --->  Minima is at = ", xk)
    print("        --->  Minimum value of Cost Function = ", f(xk))
    print("---------------------------------------------------------------\n")



# Code execution section

def main():
    x = Symbol('x')
    cost_function = input("---> Enter cost function f(x): ").strip()
    c_f = sympify(cost_function)
    f = lambdify(x, c_f, "numpy")
    Amsgrad(c_f, f)

if __name__ == "__main__":
    main()