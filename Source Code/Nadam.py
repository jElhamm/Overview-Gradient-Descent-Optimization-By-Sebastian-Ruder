# This code implements the Nadam optimization algorithm based on the cost function provided by the user.


import sys
import numpy as np
import scipy as sc
from sympy import *


def nadam(cost_function, f):
    x = Symbol('x')
    print("f(x) = ", cost_function)
    f_dash = diff(cost_function, x)
    print("df(x)/dx = ", f_dash)
    initialApproximation = float(input("\n---> Enter initial approximation: "))
    x0 = initialApproximation
    errorTolerance = float(input("---> Enter error tolerance: "))
    learningRate = float(input("---> Enter learning rate: "))
    print("\n---------------------------------------------------------------")
    print(" *** Starting Nadam")
    print("        --->  x0 = ", initialApproximation)
    print("        --->  f(x0) = ", f(initialApproximation))
 