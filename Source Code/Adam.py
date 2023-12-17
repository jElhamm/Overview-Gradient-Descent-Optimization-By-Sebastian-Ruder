# This code implements the Adam optimization algorithm based on the cost function provided by the user.


import numpy as np
import scipy as sc
from sympy import Symbol, sympify
import sys


def adam(cost_function, f):
    x = Symbol('x')
    print("f(x) = ", cost_function)
    f_derivative = diff(cost_function, x)
    print("df(x)/dx = ", f_derivative)
    initialApproximation = float(input("\n---> Enter initial approximation: "))
    x_0 = initialApproximation
    errorTolerance = float(input("---> Enter error tolerance: "))
    learningRate = float(input("---> Enter learning rate: "))
    print("\n---------------------------------------------------------------")
    print(" *** Starting Adam")
    print("        --->  x0 = ", initialApproximation)
    print("        --->  f(x0) = ", f(initialApproximation))

    #----------------------------------------------------------------------------------------------------------------------------------------------------------
    x_k = x_0
    m_0 = 0.0
    m_k = 0.0
    v_0 = 0.0
    v_k = 0.0
    b1  = 0.9
    b2  = 0.999
    x_prev  = 0.0
    epsilon = 10**-8
    iterationCount = 0
    while True:
        iterationCount += 1
        x_prev = x_0
        x_0 = x_k
        m_0 = m_k
        v_0 = v_k
        f_derivative_k = (lambdify(x, f_derivative, "numpy"))(x_k)                                    # Computing the derivative of f at x_k
        gradient_t     = f_derivative_k
        m_k  = b1 * m_0 + (1 - b1) * gradient_t                                                       # Updating m_k and v_k using exponential moving averages
        v_k  = b2 * v_0 + (1 - b2) * (gradient_t ** 2)
        mc_k = m_k / (1 - b1 ** iterationCount)                                                       # Bias correction
        vc_k = v_k / (1 - b2 ** iterationCount)
        x_k  = x_k - learningRate * mc_k / ((vc_k ** 0.5) + epsilon)                                  # Updating x_k using the Adam optimization algorithm
        if abs(N(x_k - x_0)) < errorTolerance or abs(N(x_k - x_prev)) < 0.1 * errorTolerance:         # Checking termination conditions
            break
    #----------------------------------------------------------------------------------------------------------------------------------------------------------
  
    print(" *** Number of Iterations = ", iterationCount)
    print("        --->  Minima is at = ", x_k)
    print("        --->  Minimum value of Cost Function = ", f(x_k))
    print("---------------------------------------------------------------\n")



# Code execution section
    
def main():
    x = Symbol('x')
    cost_function = input("---> Enter cost function f(x): ").strip()
    cost_function_sympy = sympify(cost_function)
    f = lambdify(x, cost_function_sympy, "numpy")
    adam(cost_function_sympy, f)

if __name__ == "__main__":
    main()  