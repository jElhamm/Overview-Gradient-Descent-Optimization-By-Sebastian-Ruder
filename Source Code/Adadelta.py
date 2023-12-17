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