# This code implements the Adadelta optimization algorithm based on the cost function provided by the user.

import numpy as np
import scipy as sc
from sympy import Symbol, sympify
import sys

def adadelta(cost_function, f):
    x = Symbol('x')
    print("f(x) = ", cost_function)
    f_dash = diff(cost_function, x)
    print("df(x)/dx = ", f_dash)
    initialApproximation = float(input("---> Enter initial approximation: "))
    x0 = initialApproximation
    decayConstant = float(input("---> Enter decay constant: "))
    errorTolerance = float(input("---> Enter error tolerance: "))
    learningRate = float(input("---> Enter learning rate: "))
    print("\n---------------------------------------------------------------")
    print(" *** Starting Adadelta")
    print("        --->  x0 = ", initialApproximation)
    print("        --->  f(x0) = ", f(initialApproximation))
    #----------------------------------------------------------------------------------------------------------------------------------------------------
    xk = x0
    eg2t_0 = 0.0
    eg2t_k = 0.0
    x_prev  = 0.0
    iter_count = 0
    e_d_theta2_0 = 0.0
    e_d_theta2_k = 0.0
    epsilon = 10.0 ** -8
    while True:
        iter_count += 1
        x_prev = x0
        x0     = xk
        eg2t_0 = eg2t_k
        e_d_theta2_0 = e_d_theta2_k
        f_k_dash     = lambdify(x, f_dash, "numpy")(xk)
        g_t          = f_k_dash
        eg2t_k       = decayConstant * eg2t_0 + (1 - decayConstant) * (g_t ** 2)                        # Update average squared gradients
        d_theta_t    = -learningRate * g_t / ((eg2t_k + epsilon) ** 0.5)                                 # Calculate change in x
        e_d_theta2_k = (decayConstant * e_d_theta2_0 + (1 - decayConstant) * (d_theta_t ** 2))          # Update average squared change in x
        xk = xk - ((e_d_theta2_0 + epsilon) ** 0.5 / (e_d_theta2_k + epsilon) ** 0.5) * g_t               # Update x
        
        print("    x" + str(iter_count) + " = ", xk)
        print("    f(x" + str(iter_count) + ") = ", f(xk))
        if abs(N(xk - x0)) < errorTolerance or abs(N(xk - x_prev)) < 0.1 * errorTolerance:
            break
    #----------------------------------------------------------------------------------------------------------------------------------------------------
    print(" *** Number of Iterations = ", iter_count)
    print("        --->  Minima is at = ", xk)
    print("        --->  Minimum value of Cost Function = ", f(xk))
    print("---------------------------------------------------------------\n")



# Code execution section

def main():
    x = Symbol('x')
    cost_function = input("---> Enter cost function f(x): ").strip()
    c_f = sympify(cost_function)
    f   = lambdify(x, c_f, "numpy")
    adadelta(c_f, f)


if __name__ == "__main__":
    main()