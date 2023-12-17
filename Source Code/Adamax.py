# This code implements the Adamax optimization algorithm based on the cost function provided by the user.

import sys
import numpy as np 
import scipy as sc 
from sympy import Symbol, diff, lambdify, sympify

def adamax(cost_function, function):
    x = Symbol('x')
    print("f(x) = ", cost_function)
    derivative = diff(cost_function, x)
    print("df(x)/dx = ", derivative)
    initialApproximation = float(input("\n---> Enter initial approximation: "))
    x0 = initialApproximation
    errorTolerance = float(input("---> Enter error tolerance: "))
    learningRate = 0.002
    print("\n---------------------------------------------------------------")
    print(" *** Starting AdaMax")
    print("        ---> x0 = ", initialApproximation)
    print("        ---> f(x0) = ", function(initialApproximation))
    