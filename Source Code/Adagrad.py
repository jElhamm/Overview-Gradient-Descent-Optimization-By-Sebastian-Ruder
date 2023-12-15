# This code implements the Adagrad optimization algorithm based on the cost function provided by the user.


import sys
import numpy as np 
import scipy as sc 
from sympy import symbols, lambdify, diff, N

def adagrad(cost_function, f):
    x = symbols('x')
    print("cost_function(x) = ", cost_function)
    f_derivative = diff(cost_function, x)
    print("df(x)/dx = ", f_derivative)
    initialApproximation = float(input("\n---> Enter initial approximation: "))
    current_x = initialApproximation
    learningRate = float(input("---> Enter learning rate: "))
    errorTolerance = float(input("---> Enter error tolerance: "))
    print("\n---------------------------------------------------------------")
    print(" *** Starting Adagrad")
    print("        --->  initial x = ", initialApproximation)
    print("        ---> cost_function(initial x) = ", f(initialApproximation))
 