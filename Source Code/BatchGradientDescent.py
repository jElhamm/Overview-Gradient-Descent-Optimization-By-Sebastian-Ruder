# This code implements the batch_Gradient_Descent optimization algorithm based on the cost function provided by the user.


import sys
import numpy as np
import scipy as sc
from sympy import Symbol, diff, sympify, lambdify


def batchGradientDescent(cost_function, f):
    x = Symbol('x')
    print("f(x) = ", cost_function)
    derivative_fx = diff(cost_function, x)
    print("df(x)/dx = ", derivative_fx)
    initialApproximation = float(input("\n---> Enter initial approximation: "))
    x0 = initialApproximation
    learningRate = float(input("---> Enter learning rate: "))
    errorTolerance = float(input("---> Enter error tolerance: "))
    print("\n---------------------------------------------------------------")
    print(" *** Starting Batch Gradient Descent")
    print("        --->  x0 =", initialApproximation)
    print("        --->  f(x0) =", f(initialApproximation))
  
  #----------------------------------------------------------------------------------------------------
    numIterations = 0
    xk = x0
    while True:
        numIterations += 1
        derivative_fk = lambdify(x, derivative_fx, "numpy")(xk)
        xk = xk - learningRate * derivative_fk
        if abs(N(xk - x0)) < errorTolerance:
            break

        x0 = xk
    #----------------------------------------------------------------------------------------------------

    print(" *** Number of Iterations =", numIterations)
    print("        --->  Minima is at =", xk)
    print("        --->  Minimum value of Cost Function =", f(xk))
    print("---------------------------------------------------------------\n")



# Code execution section

def main():
  x = Symbol('x')
  cost_function = input("---> Enter cost function f(x): ").strip()
  c_f = sympify(cost_function)
  f = lambdify(x, c_f, "numpy")
  batchGradientDescent(c_f, f)

if __name__ == "__main__":
    main()