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
  