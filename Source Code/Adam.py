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
    