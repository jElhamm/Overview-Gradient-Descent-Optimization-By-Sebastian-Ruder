# This code implements the rms_prop optimization algorithm based on the cost function provided by the user.


import sys
import numpy as np
from sympy import Symbol, diff, lambdify, sympify, N


def rms_prop(cost_function, function):
    x = Symbol('x')
    print("f(x) =", cost_function)
    function_derivative = diff(cost_function, x)
    print("df(x)/dx =", function_derivative)
    initialApproximation = float(input("\n---> Enter initial approximation: "))
    errorTolerance = float(input("---> Enter error tolerance: "))
    learningRate = float(input("---> Enter learning rate: "))
    print("\n---------------------------------------------------------------")
    print("Starting RMS Prop")
    print("        --->  x0 =", initialApproximation)
    print("        --->  f(x0) =", function(initialApproximation))
 